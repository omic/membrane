# SERVER = False
import os

import warnings
warnings.filterwarnings('ignore')

# ROOT_DIR = ""
CHECKPOINTS_FOLDER = "checkpoints"


# Data_Part
TOTAL_USERS = 100
CLIPS_PER_USER = 15 #15 #15
MIN_CLIP_DURATION = 2 #5 #2 #3
NUM_NEW_CLIPS = 2
TRAIN_PAIR_SAMPLES = None #1000

# ML_Part
DISTANCE_METRIC = "cosine"
C_THRESHOLD = THRESHOLD = 0.995 # 0.8 # similarity should be larger than
E_THRESHOLD = 2 #distance should be less than
LEARNING_RATE = 1e-3 #5e-4
N_EPOCHS = 1 #30
BATCH_SIZE = 32
TRAINING_USERS = 100
SIMILAR_PAIRS = CLIPS_PER_USER*(CLIPS_PER_USER-1)#max #None #2#20 #None for max
DISSIMILAR_PAIRS = SIMILAR_PAIRS * 5

####### For each Training data ##############
TRAIN_PATH = 'datasets/train-other-500'
STFT_FOLDER = os.path.join(TRAIN_PATH.rsplit('/')[0],'stft_{}s'.format(int(MIN_CLIP_DURATION)))
PAIRS_FILE = 'pairs_{}s.csv'.format(int(MIN_CLIP_DURATION))
CLIPS_LIST_FILE = 'clips_list.txt'
PASS_FIRST_USERS = 300

##### Augmentation ####
AUGMENT = True
SHIFT_CHANCE = 0.5 # 20% chance of shifting
W_NOISE_CHANCE = 0.8 #80% chance of white noise
NOISE_CHANCE = 0.5 # 50% chance of putting noise
BACKGROUND_LIST_PATH = 'datasets/bg_noises.txt'


####### For each Test data ###############
# TEST_STFT_FOLDER = 'omic_stft_{}s'.format(int(MIN_CLIP_DURATION))
TEST_STFT_FOLDER = 'test_stft_{}s'.format(int(MIN_CLIP_DURATION))
# TEST_PAIRS_FILE ='omic_pairs_{}s.csv'.format(int(MIN_CLIP_DURATION))
TEST_PAIRS_FILE ='test_pairs_{}s.csv'.format(int(MIN_CLIP_DURATION))
# TEST_PATH = 'datasets/omic'
TEST_PATH ='../../LibriSpeech/test-other'
# TEST_CLIPS_LIST_FILE = 'omic_clips_list'
TEST_CLIPS_LIST_FILE ='test_clips_list.txt'
TEST_CLIPS_PER_USER = None #(None means max - clips all audio files)

#recording parameters
import pyaudio
CHUNK = 1024 #1024
FORMAT = pyaudio.paInt16
try:
    CHANNELS = pyaudio.PyAudio().get_default_input_device_info()['maxInputChannels']
    #2
except:
    print("No sound channel configured. Set CHANNEL = 1")
    CHANNELS = 1
RATE = 16000 # 44100
EXTRA_SECONDS = 2.0
RECORD_SECONDS = NUM_NEW_CLIPS * MIN_CLIP_DURATION + EXTRA_SECONDS
BACKGROUND_RECORD_SECONDS = 3



# For recorder.py
# RECORDING_PATH = "recordings"
# RECORDING_STFT_FOLDER = os.path.join(RECORDING_PATH,'stft')#RECORDING_PATH + '/'+'stft'


# Files and Directories
# VGG_VOX_WEIGHT_FILE = "./vggvox_ident_net.mat"


# VBBA.py
ENROLL_RECORDING_FNAME = "enroll_recording"#.wav
VERIFY_RECORDING_FNAME = "veri_recording" #"verify_user_recording.wav"
IDENTIFY_RECORDING_FNAME = "iden_recording" #"identify_user_recording.wav"
    # MODEL_FNAME = "checkpoint_20181208-090431_0.007160770706832409.pth.tar"
SPEAKER_MODELS_FILE = 'speaker_models.pkl'
SPEAKER_PHRASES_FILE = 'speaker_phrases.pkl'
ENROLLMENT_FOLDER = "enrolled_users"
VERIFICATION_FOLDER = "tested_users"

NOISE_DURATION_FROM_FILE = 2 #(seconds)





# PAIRS_FILE = 'pairs.csv'


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
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.manifold import TSNE
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support as score

import librosa
import librosa.display
import speech_recognition as sr
# import pyaudio
import wave
import contextlib
# import matplotlib.pyplot as plt
# %matplotlib inline
# import seaborn as sns

#Voice-to-text
from difflib import SequenceMatcher
from deepspeech import Model#, printVersions

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torch.utils.checkpoint import checkpoint


# plt.style.use('seaborn-darkgrid')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



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


# Neural Network parameters
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


#####################Voice-To-Text##############

#######Deepspeech Voice-To-Text Parameters########

DS_model_file_path = 'deepspeech_data/deepspeech-0.7.0-models.pbmm'
beam_width = 500
DS_model = Model(DS_model_file_path)
DS_model.setBeamWidth(beam_width)
DS_model.enableExternalScorer('deepspeech_data/deepspeech-0.7.0-models.scorer')


def get_text(data, model = DS_model):
#     y , s = librosa.load(fpath, sr=16000)
    y = (data* 32767).astype('int16')
    text = model.stt(y)
    return text

def get_text_score(phrase1, phrase2):
    return SequenceMatcher(a= phrase1, b= phrase2).ratio()


#############Voice-Recognition, Recording############

def get_stft(all_x, nperseg=400, noverlap=239, nfft=1023):
    all_stft = []
    for x in all_x:
        _, t, Z = scipy.signal.stft(x, window="hamming",
                                       nperseg=nperseg,
                                       noverlap=noverlap,
                                       nfft=nfft)
        Z = sklearn.preprocessing.normalize(np.abs(Z), axis=1)
        assert Z.shape[0] == 512
        all_stft.append(Z)
    return np.array(all_stft)


def split_recording(recording=ENROLL_RECORDING_FNAME):
#     wav, sr = librosa.load(recording)
    RECORD_SECONDS = int(NUM_NEW_CLIPS * MIN_CLIP_DURATION)
    all_x = []
    for offset in range(0, RECORD_SECONDS, int(MIN_CLIP_DURATION)):
        x, sr = librosa.load(recording, sr=16000, offset=offset,
                             duration=MIN_CLIP_DURATION)
        all_x.append(x)
    return get_stft(all_x)

def split_loaded_data(data, sr = RATE):
    RECORD_SECONDS = int(NUM_NEW_CLIPS * MIN_CLIP_DURATION)
    RECORD_SECONDS = int(min(RECORD_SECONDS, len(data)/sr))
    all_x = []
    for offset in range(0, RECORD_SECONDS, int(MIN_CLIP_DURATION)):
        x = data[offset:offset+MIN_CLIP_DURATION*sr]
        all_x.append(x)
    return get_stft(all_x)

# denoising functions

def _stft(x, nperseg=400, noverlap=239, nfft=1023):
    _, _, Z = scipy.signal.stft(x, window="hamming",
                                   nperseg=nperseg,
                                   noverlap=noverlap,
                                   nfft=nfft)
    assert Z.shape[0] == 512
    return np.array(Z)


def _istft(x, nperseg=400, noverlap=239, nfft=1023):

    _, Z = scipy.signal.istft(x, window="hamming",
                                   nperseg=nperseg,
                                   noverlap=noverlap,
                                   nfft=nfft)
    return np.array(Z)

def _amp_to_db(x):
    return librosa.core.amplitude_to_db(x, ref=1.0, amin=1e-20, top_db=80.0)

def _db_to_amp(x,):
    return librosa.core.db_to_amplitude(x, ref=1.0)

def removeNoise(
    audio_data,
    noise_data,
    #nperseg=400, noverlap=239, nfft=1023
    n_grad_freq=2,
    n_grad_time=4,
#     n_fft=2048,
#     n_fft=1023,
#     win_length=2048,
#     hop_length=512,
    n_std_thresh=1.5,
    prop_decrease=1.0
):
    """Remove noise from audio based upon a clip containing only noise

    Args:
        audio_data (array): The first parameter.
        noise_data (array): The second parameter.
        n_grad_freq (int): how many frequency channels to smooth over with the mask.
        n_grad_time (int): how many time channels to smooth over with the mask.
        n_fft (int): number audio of frames between STFT columns.
        win_length (int): Each frame of audio is windowed by `window()`. The window will be of length `win_length` and then padded with zeros to match `n_fft`..
        hop_length (int):number audio of frames between STFT columns.
        n_std_thresh (int): how many standard deviations louder than the mean dB of the noise (at each frequency level) to be considered signal
        prop_decrease (float): To what extent should you decrease noise (1 = all, 0 = none)
        visual (bool): Whether to plot the steps of the algorithm

    Returns:
        array: The recovered signal with noise subtracted

    """

    noise_stft = _stft(noise_data)
    noise_stft_db = _amp_to_db(np.abs(noise_stft))
    mean_freq_noise = np.mean(noise_stft_db, axis=1)
    std_freq_noise = np.std(noise_stft_db, axis=1)
    noise_thresh = mean_freq_noise + std_freq_noise * n_std_thresh

    sig_stft = _stft(audio_data)
    sig_stft_db = _amp_to_db(np.abs(sig_stft))

    mask_gain_dB = np.min(sig_stft_db)

    filter_compt = np.concatenate(
            [
                np.linspace(0, 1, n_grad_freq + 1, endpoint=False),
                np.linspace(1, 0, n_grad_freq + 2),
            ]
        )[1:-1]

    smoothing_filter = np.outer(
            filter_compt,
            filter_compt,
        )
    smoothing_filter = smoothing_filter / np.sum(smoothing_filter)
    # calculate the threshold for each frequency/time bin
    db_thresh = np.repeat(
        np.reshape(noise_thresh, [1, len(mean_freq_noise)]),
        np.shape(sig_stft_db)[1],
        axis=0,
    ).T

    sig_mask = sig_stft_db < db_thresh

    sig_mask = scipy.signal.fftconvolve(sig_mask, smoothing_filter, mode="same")
    sig_mask = sig_mask * prop_decrease

    sig_stft_db_masked = (
        sig_stft_db * (1 - sig_mask)
        + np.ones(np.shape(mask_gain_dB)) * mask_gain_dB * sig_mask
    )  # mask real
    sig_imag_masked = np.imag(sig_stft) * (1 - sig_mask)
    sig_stft_amp = (_db_to_amp(sig_stft_db_masked) * np.sign(sig_stft)) + (
        1j * sig_imag_masked
    )

    recovered_signal = _istft(sig_stft_amp)
    recovered_spec = _amp_to_db(
        np.abs(_stft(recovered_signal)))

    return recovered_signal.astype('float32')


def write_recording(fpath, audio_data):
    librosa.output.write_wav(fpath+'.wav', audio_data, sr=RATE)


def fpath_numbering(fpath, extension = '.wav'):
    while os.path.exists(fpath+extension):
        if fpath[-1].isalpha():
            fpath = fpath+'2'
        else:
            fpath = fpath[:-1]+str(int(fpath[-1])+1)
    return fpath




