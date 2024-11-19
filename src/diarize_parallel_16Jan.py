import argparse
import os
# from helpers import *
from faster_whisper import WhisperModel
import whisperx
import torch
from deepmultilingualpunctuation import PunctuationModel
import re
import subprocess
import logging
import time
from datetime import timedelta
import contractions
from src import *


torch.set_num_threads(6)

mtypes = {"cpu": "int8", "cuda": "int8"}
stemming = False
device = "cpu"
model_name = "medium.en"
suppress_numerals = False
def init_models():
    whisper_model = WhisperModel(
        model_name, device="cuda", compute_type=mtypes[device]
    )
    punct_model = PunctuationModel(model="kredor/punctuate-all")
    return whisper_model, punct_model

def process(audio_path,whisper_model,punct_model):
    startTime = time.time()

    if stemming:
        # Isolate vocals from the rest of the audio

        return_code = os.system(
            f'python3 -m demucs.separate -n htdemucs --two-stems=vocals "{audio_path}" -o "temp_outputs"'
        )

        if return_code != 0:
            logging.warning(
                "Source splitting failed, using original audio file. Use --no-stem argument to disable it."
            )
            vocal_target = audio_path
        else:
            vocal_target = os.path.join(
                "temp_outputs", "htdemucs", os.path.basename(audio_path[:-4]), "vocals.wav"
            )
    else:
        vocal_target = audio_path
    
    
    logging.info("Starting Nemo process with vocal_target: ", vocal_target)
    nemo_process = subprocess.Popen(
        ["python3", "nemo_process.py", "-a", vocal_target, "--device", "cpu"],
    )
    
    # Run on GPU with FP16

    # or run on GPU with INT8
    # model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
    # or run on CPU with INT8
    # model = WhisperModel(model_size, device="cpu", compute_type="int8")

    if suppress_numerals:
        numeral_symbol_tokens = find_numeral_symbol_tokens(whisper_model.hf_tokenizer)
    else:
        numeral_symbol_tokens = None

    segments, info = whisper_model.transcribe(
        vocal_target,
        beam_size=5,
        word_timestamps=True,
        suppress_tokens=numeral_symbol_tokens,
        vad_filter=True,
    )
    whisper_results = []
    
    forTimeStart = time.time()
    for segment in segments:
        whisper_results.append(segment._asdict())
    forTimeEnd = time.time()
    print("Loop time:", str(timedelta(seconds=forTimeEnd - forTimeStart)))
    
    # clear gpu vram
    # del whisper_model
    # torch.cuda.empty_cache()

    if info.language in wav2vec2_langs:
        alignment_model, metadata = whisperx.load_align_model(
            language_code=info.language, device=device
        )
        result_aligned = whisperx.align(
            whisper_results, alignment_model, metadata, vocal_target, device
        )
        word_timestamps = filter_missing_timestamps(result_aligned["word_segments"])
        # clear gpu vram
        del alignment_model
        # torch.cuda.empty_cache()
    else:
        word_timestamps = []
        for segment in whisper_results:
            for word in segment["words"]:
                word_timestamps.append({"word": word[2], "start": word[0], "end": word[1]})

    # Reading timestamps <> Speaker Labels mapping
    nemoTime = time.time()
    nemo_process.communicate()
    nemoTimeEnd = time.time()
    print("Nemo time:", str(timedelta(nemoTimeEnd - nemoTime)))
    ROOT = os.getcwd()
    temp_path = os.path.join(ROOT, "temp_outputs")

    speaker_ts = []
    with open(os.path.join(temp_path, "pred_rttms", "mono_file.rttm"), "r") as f:
        lines = f.readlines()
        for line in lines:
            line_list = line.split(" ")
            s = int(float(line_list[5]) * 1000)
            e = s + int(float(line_list[8]) * 1000)
            speaker_ts.append([s, e, int(line_list[11].split("_")[-1])])

    wsm = get_words_speaker_mapping(word_timestamps, speaker_ts, "start")

    if info.language in punct_model_langs:
        # restoring punctuation in the transcript to help realign the sentences
        # punct_model = PunctuationModel(model="kredor/punctuate-all")

        words_list = list(map(lambda x: x["word"], wsm))

        labled_words = punct_model.predict(words_list)

        ending_puncts = ".?!"
        model_puncts = ".,;:!?"

        # We don't want to punctuate U.S.A. with a period. Right?
        is_acronym = lambda x: re.fullmatch(r"\b(?:[a-zA-Z]\.){2,}", x)

        for word_dict, labeled_tuple in zip(wsm, labled_words):
            word = word_dict["word"]
            if (
                word
                and labeled_tuple[1] in ending_puncts
                and (word[-1] not in model_puncts or is_acronym(word))
            ):
                word += labeled_tuple[1]
                if word.endswith(".."):
                    word = word.rstrip(".")
                word_dict["word"] = word

        wsm = get_realigned_ws_mapping_with_punctuation(wsm)
    else:
        logging.warning(
            f"Punctuation restoration is not available for {info.language} language."
        )

    ssm = get_sentences_speaker_mapping(wsm, speaker_ts)

    # with open(f"{audio_path[:-4]}.txt", "w", encoding="utf-8-sig") as f:
    #     get_speaker_aware_transcript(ssm, f)

    # with open(f"{audio_path[:-4]}.srt", "w", encoding="utf-8-sig") as srt:
    #     write_srt(ssm, srt)
    final_string = ''
    previous = ''
    for details in ssm:
        curr = details['speaker']
        if previous != curr:
            final_string += details['speaker'] + ": " + details['text']
            # final_output.append(f"{details['speaker']} : {details['text']}")
        else:
            final_string += details['text']
        previous = curr
    expanded_string = []
    for word in final_string.split():
        expanded_string.append(contractions.fix(word))
    endTime = time.time()
    print("Time taken:", str(timedelta(seconds=endTime - startTime)))
    cleanup(temp_path)
   
    return " ".join(expanded_string)

    # end_time = time.time()
    # time_taken = end_time - start_time

    # print("Time taken:", str(timedelta(seconds=time_taken)))

