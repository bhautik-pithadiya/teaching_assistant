from src import diarize

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


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)




if __name__ == "__main__":
    whis,msdd,pun = diarize.init_models()
    
    audiopath = "./WhatsApp Audio 2024-12-03 at 9.45.31 PM.mp3"   
    print(diarize.process(audio_path=audiopath,whisper_model=whis,msdd_model=msdd,punct_model=pun))