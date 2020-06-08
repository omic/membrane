SERVER = False
import os


# Files and Directories
ROOT_DIR = ""
TRAIN_PATH = 'wav_train_subset'
TEST_PATH = 'wav_test_subset'
CHECKPOINTS_FOLDER = "checkpoints"
VGG_VOX_WEIGHT_FILE = "./vggvox_ident_net.mat"

RECORDING_PATH = "recordings"
ENROLL_RECORDING_FNAME = "enroll_user_recording.wav"
VERIFY_RECORDING_FNAME = "verify_user_recording.wav"
IDENTIFY_RECORDING_FNAME = "identify_user_recording.wav"
# MODEL_FNAME = "checkpoint_20181208-090431_0.007160770706832409.pth.tar"
SPEAKER_MODELS_FILE = 'speaker_models.pkl'
ENROLLMENT_FOLDER = "enrolled_users"
VERIFICATION_FOLDER = "tested_users"

# Data_Part
TOTAL_USERS = 100
CLIPS_PER_USER = 15 #15
MIN_CLIP_DURATION = 3. #5
NUM_NEW_CLIPS = 5

# ML_Part
TRAINING_USERS = 100
SIMILAR_PAIRS = 20#20
DISSIMILAR_PAIRS = SIMILAR_PAIRS * 5
DISTANCE_METRIC = "cosine"
THRESHOLD = 0.95 # 0.8

LEARNING_RATE = 5e-4
N_EPOCHS = 30
BATCH_SIZE = 32

STFT_FOLDER = 'stft' + str(int(MIN_CLIP_DURATION))
TEST_STFT_FOLDER = 'test_stft' + str(int(MIN_CLIP_DURATION))
RECORDING_STFT_FOLDER = os.path.join(RECORDING_PATH,'stft')#RECORDING_PATH + '/'+'stft'

PAIRS_FILE = 'pairs_{}s.csv'.format(int(MIN_CLIP_DURATION))
# PAIRS_FILE = 'pairs.csv'
TEST_PAIRS_FILE = 'test_pairs.csv'

assert SIMILAR_PAIRS <= CLIPS_PER_USER * (CLIPS_PER_USER - 1)


from tqdm import tqdm

import sys
import time
try:
    import cPickle as pickle
except:
    import pickle
import itertools
from collections import Counter
from collections import OrderedDict
from IPython.core.display import HTML
import argparse

import numpy as np
import pandas as pd
from scipy.io import loadmat
import scipy
import sklearn
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support as score

import librosa
import librosa.display
import speech_recognition as sr
import pyaudio
import wave
import contextlib
import matplotlib.pyplot as plt

# import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torch.utils.checkpoint import checkpoint

if not os.path.exists(STFT_FOLDER):
    os.mkdir(STFT_FOLDER)

if not os.path.exists(CHECKPOINTS_FOLDER):
    os.mkdir(CHECKPOINTS_FOLDER)

if not os.path.exists(ENROLLMENT_FOLDER):
    os.mkdir(ENROLLMENT_FOLDER)

if not os.path.exists(VERIFICATION_FOLDER):
    os.mkdir(VERIFICATION_FOLDER)

plt.style.use('seaborn-darkgrid')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_rel_path(path, server=SERVER, root_dir=ROOT_DIR):
    if server:
        return os.path.join(root_dir, path)
    else:
        return path


def wavPlayer(filepath):
    src = """
    <head>
    <meta http-equiv="Content-Type" content="text/html; charset=utf-8">
    <title>Simple Test</title>
    </head>

    <body>
    <audio controls="controls" style="width:600px" >
      <source src="%s" type="audio/wav" />
      Your browser does not support the audio element.
    </audio>
    </body>
    """%(filepath)
    display(HTML(src))


def get_waveform(clip_list, offset=0., duration=MIN_CLIP_DURATION):
    all_x = []
    all_sr = []
    for path in tqdm(clip_list):
        x, sr = librosa.load(path, sr=None, offset=offset,
                             duration=duration)
        all_x.append(x)
        all_sr.append(sr)

    assert len(np.unique(np.array(all_sr))) == 1
    return all_x, all_sr


def get_stft(all_x, nperseg=400, noverlap=239, nfft=1023):

    all_stft = []
    for x in all_x:
        _, _, Z = scipy.signal.stft(x, window="hamming",
                                       nperseg=nperseg,
                                       noverlap=noverlap,
                                       nfft=nfft)
        Z = sklearn.preprocessing.normalize(np.abs(Z), axis=1)
        assert Z.shape[0] == 512
        all_stft.append(Z)
    return np.array(all_stft)


def load_pretrained_weights():
    weights = {}

    # loading pretrained vog_vgg learned weights
    vox_weights = loadmat(get_rel_path(VGG_VOX_WEIGHT_FILE),
                          struct_as_record=False, squeeze_me=True)

    for l in vox_weights['net'].layers[:-1]:
        if len(l.weights) > 0:
            weights[l.name] = l.weights
    #         print(l.name, [i.shape for i in l.weights])

    for i in weights:
        weights[i][0] = weights[i][0].T

    weights['conv1'][0] = np.expand_dims(weights['conv1'][0], axis=1)
    weights['fc6'][0] = np.expand_dims(weights['fc6'][0], axis=3)
    weights['fc7'][0] = np.expand_dims(weights['fc7'][0], axis=-1)
    weights['fc7'][0] = np.expand_dims(weights['fc7'][0], axis=-1)

#     print(weights.keys())
#     for key in weights:
#         print(key, [i.shape for i in weights[key]])
    return weights


# parameters
conv_kernel1, n_f1, s1, p1 = 7, 96, 2, 1
pool_kernel1, pool_s1 = 3, 2

conv_kernel2, n_f2, s2, p2 = 5, 256, 2, 1
pool_kernel2, pool_s2 = 3, 2

conv_kernel3, n_f3, s3, p3 = 3, 384, 1, 1

conv_kernel4, n_f4, s4, p4 = 3, 256, 1, 1

conv_kernel5, n_f5, s5, p5 = 3, 256, 1, 1
pool_kernel5_x, pool_kernel5_y, pool_s5_x, pool_s5_y = 5, 3, 3, 2

conv_kernel6_x, conv_kernel6_y, n_f6, s6 = 9, 1, 4096, 1

conv_kernel7, n_f7, s7 = 1, 1024, 1

conv_kernel8, n_f8, s8 = 1, 1024, 1


def save_checkpoint(state, loss):
    """Save checkpoint if a new best is achieved"""
    fname = "checkpoint_" + time.strftime("%Y%m%d-%H%M%S") + "_" + str(loss.item()) + ".pth.tar"
    torch.save(state, get_rel_path(os.path.join(CHECKPOINTS_FOLDER, fname)))  # save checkpoint
    print("$$$ Saved a new checkpoint\n")


def record(fpath):
    CHUNK = 2048 #1024
    FORMAT = pyaudio.paInt16
    CHANNELS = pyaudio.PyAudio().get_default_input_device_info()['maxInputChannels'] #2
    RATE = 16000 # 44100
    EXTRA_SECONDS = 2.0
    RECORD_SECONDS = NUM_NEW_CLIPS * MIN_CLIP_DURATION + EXTRA_SECONDS

    LONG_STRING = "She had your dark suit in greasy wash water all year. Don't ask me to carry an oily rag like that!"

    print("Recording {} seconds".format(RECORD_SECONDS - EXTRA_SECONDS))
    print("\n Speak the following sentence for recording: \n {} \n".format(LONG_STRING))

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                    input=True, frames_per_buffer=CHUNK)

    time.sleep(1)

    print("Recording starts in 3 seconds...")
    time.sleep(2)   # start 1 second earlier
    print("Speak now!")
    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK, exception_on_overflow = False)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()

    print("Recording complete")
    
    #while os.path.exists(fpath)
    wf = wave.open(fpath, 'wb') 
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()




def split_recording(recording=ENROLL_RECORDING_FNAME):
    wav, sr = librosa.load(recording)
    RECORD_SECONDS = int(NUM_NEW_CLIPS * MIN_CLIP_DURATION)
    all_x = []
    for offset in range(0, RECORD_SECONDS, int(MIN_CLIP_DURATION)):
        x, sr = librosa.load(recording, sr=16000, offset=offset,
                             duration=MIN_CLIP_DURATION)

        all_x.append(x)

    return get_stft(all_x)

def save_stft(all_stft, recording=ENROLL_RECORDING_FNAME):
    all_stft_paths = []
    for i in tqdm(range(len(all_stft))):
        user_stft = all_stft[i]
        stft_fname = '_'.join(recording.split('/')[-3:])[:-4] + '.npy'
        stft_path = get_rel_path(os.path.join(RECORDING_STFT_FOLDER, stft_fname))
        np.save(stft_path, user_stft)
        all_stft_paths.append(stft_path)
    return all_stft_paths


class AudioRec(object):
    def __init__(self):
        self.r = sr.Recognizer()
        self.src = sr.Microphone()
        with self.src as source:
            print("Calibrating microphone...")
            self.r.adjust_for_ambient_noise(source, duration=2)

    def listen(self, save_path):
        time_to_record = NUM_NEW_CLIPS * MIN_CLIP_DURATION + 1.0
        with self.src as source:
            print("Recording ...", time_to_record)
            # record for a maximum of 10s
            audio = self.r.listen(source, phrase_time_limit=time_to_record)
        # write audio to a WAV file
        with open(save_path, "wb") as f:
            f.write(audio.get_wav_data())

