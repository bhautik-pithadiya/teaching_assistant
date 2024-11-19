import argparse
import os
from src import *
import torch
from pydub import AudioSegment
from nemo.collections.asr.models.msdd_models import NeuralDiarizer
import time
from datetime import timedelta

parser = argparse.ArgumentParser()
parser.add_argument(
    "-a", "--audio", help="name of the target audio file", required=True
)
parser.add_argument(
    "--device",
    dest="device",
    default="cuda" if torch.cuda.is_available() else "cpu",
    help="if you have a GPU use 'cuda', otherwise 'cpu'",
)
args = parser.parse_args()
torch.set_num_threads(8)
# convert audio to mono for NeMo combatibility
startConvert = time.time()
sound = AudioSegment.from_file(args.audio).set_channels(1)
ROOT = os.getcwd()
temp_path = os.path.join(ROOT, "temp_outputs")
os.makedirs(temp_path, exist_ok=True)
sound.export(os.path.join(temp_path, "mono_file.wav"), format="wav")
endConvert = time.time()
print("Time for convert:", str(timedelta(seconds=endConvert - startConvert)))

# Initialize NeMo MSDD diarization model
msdd_model = NeuralDiarizer(cfg=create_config(temp_path)).to(args.device)
msdd_model.diarize()
endMSDD = time.time()
print("SD TIme:", str(timedelta(seconds=endMSDD - endConvert)))
