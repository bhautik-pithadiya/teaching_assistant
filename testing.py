from src import diarize
from src import *
import time
from datetime import timedelta
import os
import wget
import json
import shutil
from faster_whisper import WhisperModel
import whisperx
import torch
from pydub import AudioSegment
from nemo.collections.asr.models.msdd_models import NeuralDiarizer
from deepmultilingualpunctuation import PunctuationModel
import re
from src import *
# from . import *
import time
from datetime import timedelta
from numba import cuda
import logging
import contractions
import concurrent.futures
import multiprocessing
from multiprocessing.pool import ThreadPool
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from itertools import islice
from numba import jit
import torchaudio
from multiprocessing import Pool
# "/home/ksuser/Documents/conversationTranscribe/data/input_audio/1696528084008_1000022968016_1022_2224792_100.mp3"


path = "./1696528151059_1000050599709_1028_2224792.mp3"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Name of the audio file ---> Change it to folder path containing multiple audio files.
#audio_path = "/home/ksuser/LS/APAK.ai-main/audio_files/1696528455061_1000085836312_1012_2224792.mp3"

# torch.set_num_threads(5)

# Whether to enable music removal from speech, helps increase diarization quality but uses alot of ram
enable_stemming = False

# (choose from 'tiny.en', 'tiny', 'base.en', 'base', 'small.en', 'small', 'medium.en', 'medium', 'large-v1', 'large-v2', 'large')
whisper_model_name = "medium.en"

# replaces numerical digits with their pronounciation, increases diarization accuracy 
suppress_numerals = False 

#models

def init_models():
    ROOT = os.getcwd()
    temp_path = os.path.join(ROOT, "temp_outputs")
    os.makedirs(temp_path, exist_ok=True)
    whisper_model = WhisperModel(whisper_model_name, device="cuda", compute_type="float16")
    msdd_model = NeuralDiarizer(cfg=create_config(temp_path)).to("cuda")
    punct_model = PunctuationModel(model="kredor/punctuate-all")
    return whisper_model, msdd_model, punct_model

def load_audio(audio_path):
    waveform, sample_rate = torchaudio.load(audio_path)
    return waveform.to(device), sample_rate


def transcribe_on_gpu(model, audio_path, **kwargs):
    # Load and move audio data to GPU
    waveform, sample_rate = load_audio(audio_path)
    
    # Perform transcription
    segments, _ = model.transcribe(waveform, **kwargs)
    
    # Process segments (this part runs on CPU, ensure efficient transfer)
    whisper_results = [segment._asdict() for segment in segments]
    
    return whisper_results

def process(audio_path, whisper_model, msdd_model, punct_model):
    
    vocal_target = audio_path
    startTime = time.time()
    
    if suppress_numerals:
        numeral_symbol_tokens = find_numeral_symbol_tokens(whisper_model.hf_tokenizer)
    else:
        numeral_symbol_tokens = None
        
    startTime1 = time.time()
    
    whisper_results = transcribe_on_gpu(
        whisper_model,
        vocal_target,
        beam_size=5,
        word_timestamps=False,
        suppress_tokens=numeral_symbol_tokens,
        vad_filter=False,)
    
    # print(whisper_results)
    segments, info = whisper_model.transcribe(
        vocal_target,
        beam_size=5,
        word_timestamps=False,
        suppress_tokens=numeral_symbol_tokens,
        vad_filter=False,
    )
    
    logger.info('       Out of Whisper.transcribe')
    whisper_results = []
    toal_info = []
    
    start = time.time()
    for segment in segments:
        whisper_results.append(segment._asdict())        
    end = time.time()
    print('Total time in loop - ' ,str(timedelta(seconds= end - start)))      
        
            
    if info.language in wav2vec2_langs:
        device = "cuda"
        alignment_model, metadata = whisperx.load_align_model(
            language_code=info.language, device=device
        )
        result_aligned = whisperx.align(
            whisper_results, alignment_model, metadata, vocal_target, device
        )
        word_timestamps = filter_missing_timestamps(result_aligned["word_segments"])

        # clear gpu vram
        # del alignment_model
        # torch.cuda.empty_cache()
    else:
        word_timestamps = []
        for segment in whisper_results:
            for word in segment["words"]:
                word_timestamps.append({"word": word[2], "start": word[0], "end": word[1]})

    print('out of if else')
    sound = AudioSegment.from_file(vocal_target).set_channels(1)
    
    ROOT = os.getcwd()
    temp_path = os.path.join(ROOT, "temp_outputs")
    os.makedirs(temp_path, exist_ok=True)
    sound.export(os.path.join(temp_path, "mono_file.wav"), format="wav")
    os.system( f'cp src/nemo_msdd_configs/diar_infer_telephonic.yaml "{temp_path}"')

    # Initialize NeMo MSDD diarization model
    # msdd_model = NeuralDiarizer(cfg=create_config(temp_path))

    msdd_model.diarize()
    
    
    # del msdd_model
    # torch.cuda.empty_cache()
    # Reading timestamps <> Speaker Labels mapping

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
        print(
            f'Punctuation restoration is not available for {whisper_results["language"]} language.'
        )

    ssm = get_sentences_speaker_mapping(wsm, speaker_ts)

    final_string = ''
    previous = ''
    for details in ssm:
        curr = details['speaker']
        if previous != curr:
            final_string += details['speaker'] + ": " + details['text']
        else:
            final_string += details['text']
        previous = curr
    expanded_string = []
    for word in final_string.split():
        expanded_string.append(contractions.fix(word))
    endTime = time.time()
    print("Time taken:", str(timedelta(seconds=endTime - startTime)))
    
    return " ".join(expanded_string)

    
    
if __name__ == "__main__":
    whis,msdd,pun = init_models()
    
    audiopath = '1696528151059_1000050599709_1028_2224792.mp3'
    
    print(process(audio_path=audiopath,whisper_model=whis,msdd_model=msdd,punct_model=pun))
   
