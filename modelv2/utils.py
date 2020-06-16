SERVER = False
import os

ROOT_DIR = ""
CHECKPOINTS_FOLDER = "checkpoints"


# Data_Part
TOTAL_USERS = 100
CLIPS_PER_USER = 15 #15 #15
MIN_CLIP_DURATION = 2 #5 #2 #3
NUM_NEW_CLIPS = 5
TRAIN_PAIR_SAMPLES = None #1000

# ML_Part
DISTANCE_METRIC = "cosine"
THRESHOLD = 0.95 # 0.8
LEARNING_RATE = 1e-3 #5e-4
N_EPOCHS = 1 #30
BATCH_SIZE = 32
TRAINING_USERS = 100
SIMILAR_PAIRS = 2#20
DISSIMILAR_PAIRS = SIMILAR_PAIRS * 5


####### For both Training, Test ##########
def find_username(fpath):
    i = fpath.rfind('/')
    return fpath[i+1:fpath.find('-', i)]
#     return fpath[i+1:fpath.find('_', i)]


####### For each Training data ##############
TRAIN_PATH = 'datasets/train-other-500'
STFT_FOLDER = os.path.join(TRAIN_PATH.rsplit('/')[0],'stft_{}s'.format(int(MIN_CLIP_DURATION)))
PAIRS_FILE = 'pairs_{}s.csv'.format(int(MIN_CLIP_DURATION))
CLIPS_LIST_FILE = 'clips_list.txt'
PASS_FIRST_USERS = 100

##### Augmentation ####
AUGMENT = True
SHIFT_CHANCE = 0.5 # 20% chance of shifting
W_NOISE_CHANCE = 0.8 #80% chance of white noise
NOISE_CHANCE = 0.5 # 50% chance of putting noise
BACKGROUND_LIST_PATH = 'datasets/bg_noises.txt'


####### For each Test data ###############
TEST_STFT_FOLDER = 'test_stft_{}s'.format(int(MIN_CLIP_DURATION))
TEST_PAIRS_FILE = 'test_pairs_{}s.csv'.format(int(MIN_CLIP_DURATION))
TEST_PATH = 'wav_test_subset'
TEST_CLIPS_LIST_FILE = 'test_clips_list.txt'

# For recorder.py
RECORDING_PATH = "recordings"
RECORDING_STFT_FOLDER = os.path.join(RECORDING_PATH,'stft')#RECORDING_PATH + '/'+'stft'


# Files and Directories
VGG_VOX_WEIGHT_FILE = "./vggvox_ident_net.mat"


# VBBA.py
ENROLL_RECORDING_FNAME = "enroll_recording"#.wav
VERIFY_RECORDING_FNAME = "veri_recording" #"verify_user_recording.wav"
IDENTIFY_RECORDING_FNAME = "iden_recording" #"identify_user_recording.wav"
    # MODEL_FNAME = "checkpoint_20181208-090431_0.007160770706832409.pth.tar"
SPEAKER_MODELS_FILE = 'speaker_models.pkl'
ENROLLMENT_FOLDER = "enrolled_users"
VERIFICATION_FOLDER = "tested_users"

NOISE_DURATION_FROM_FILE = 4 #(seconds)







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

# in dataprepocessing
# def get_waveform(clip_list, offset=0., duration=MIN_CLIP_DURATION):
#     all_x = []
#     all_sr = []
#     for path in tqdm(clip_list):
#         x, sr = librosa.load(path, sr=None, offset=offset,
#                              duration=duration)
#         all_x.append(x)
#         all_sr.append(sr)

#     assert len(np.unique(np.array(all_sr))) == 1
#     return all_x, all_sr




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


def record(fpath, enroll = False):
    CHUNK = 1024 #2048 #1024
    FORMAT = pyaudio.paInt16
    CHANNELS = pyaudio.PyAudio().get_default_input_device_info()['maxInputChannels'] #2
    RATE = 16000 # 44100
    EXTRA_SECONDS = 2.0
    RECORD_SECONDS = NUM_NEW_CLIPS * MIN_CLIP_DURATION + EXTRA_SECONDS

    LONG_STRING = "  \"She had your dark suit in greasy wash water all year. Don't ask me to carry an oily rag like that!\""

    print("Recording {} seconds".format(RECORD_SECONDS - EXTRA_SECONDS))
    print("\n Speak the following sentence for recording: \n {}\n".format(LONG_STRING))

    print(' or\n')
    print(' You can speak your own secret phrases.')
    if enroll:
        print(' If you do so, please let us know your secret phrases:)\n\n')
    else: print('\n')
    
    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                    input=True, frames_per_buffer=CHUNK)

    if enroll:input('Ready to start? (press enter)')
    else: time.sleep(1)

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
    

    wf = wave.open(fpath, 'wb') 
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    
    

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

def split_loaded_data(data, sr = 16000):
    RECORD_SECONDS = int(NUM_NEW_CLIPS * MIN_CLIP_DURATION)
    all_x = []
    for offset in range(0, RECORD_SECONDS, int(MIN_CLIP_DURATION)):
        x = data[offset:offset+MIN_CLIP_DURATION*sr]
        all_x.append(x)
    return get_stft(all_x)


# def save_stft(all_stft, audio_path=ENROLL_RECORDING_FNAME, stft_path = STFT_FOLDER):
#     all_stft_paths = []
#     for i in tqdm(range(len(all_stft))):
#         user_stft = all_stft[i]
#         stft_fname = '_'.join(audio_path.split('/')[-3:])[:-4] + '.npy'
#         stft_path = get_rel_path(os.path.join(stft_path, stft_fname))
#         np.save(stft_path, user_stft)
#         all_stft_paths.append(stft_path)
#     return all_stft_paths

# moved to realtime.py
# class AudioRec(object):
#     def __init__(self):
#         self.r = sr.Recognizer()
#         self.src = sr.Microphone()
#         with self.src as source:
#             print("Calibrating microphone...")
#             self.r.adjust_for_ambient_noise(source, duration=2)

#     def listen(self, save_path):
#         time_to_record = NUM_NEW_CLIPS * MIN_CLIP_DURATION + 1.0
#         with self.src as source:
#             print("Recording ...", time_to_record)
#             # record for a maximum of 10s
#             audio = self.r.listen(source, phrase_time_limit=time_to_record)
#         # write audio to a WAV file
#         with open(save_path, "wb") as f:
#             f.write(audio.get_wav_data())



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


# inputs: data after librosa.load('....wav', sr=16000)
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
#     if verbose:
#         start = time.time()
    # STFT over noise
    noise_stft = _stft(noise_data)
    noise_stft_db = _amp_to_db(np.abs(noise_stft))  # convert to dB
    # Calculate statistics over noise
    mean_freq_noise = np.mean(noise_stft_db, axis=1)
    std_freq_noise = np.std(noise_stft_db, axis=1)
    noise_thresh = mean_freq_noise + std_freq_noise * n_std_thresh
#     if verbose:
#         print("STFT on noise:", td(seconds=time.time() - start))
#         start = time.time()
    # STFT over signal
#     if verbose:
#         start = time.time()
    sig_stft = _stft(audio_data)
    sig_stft_db = _amp_to_db(np.abs(sig_stft))
#     if verbose:
#         print("STFT on signal:", td(seconds=time.time() - start))
#         start = time.time()
    # Calculate value to mask dB to
#     mask_gain_dB = np.min(_amp_to_db(np.abs(sig_stft)))
    mask_gain_dB = np.min(sig_stft_db)
#     print("Noise threshold, Mask gain dB:\n",noise_thresh, mask_gain_dB)
    # Create a smoothing filter for the mask in time and frequency
    
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
    # mask if the signal is above the threshold
    sig_mask = sig_stft_db < db_thresh
#     if verbose:
#         print("Masking:", td(seconds=time.time() - start))
#         start = time.time()
    # convolve the mask with a smoothing filter
    sig_mask = scipy.signal.fftconvolve(sig_mask, smoothing_filter, mode="same")
    sig_mask = sig_mask * prop_decrease
#     if verbose:
#         print("Mask convolution:", td(seconds=time.time() - start))
#         start = time.time()
    # mask the signal
    sig_stft_db_masked = (
        sig_stft_db * (1 - sig_mask)
        + np.ones(np.shape(mask_gain_dB)) * mask_gain_dB * sig_mask
    )  # mask real
    sig_imag_masked = np.imag(sig_stft) * (1 - sig_mask)
    sig_stft_amp = (_db_to_amp(sig_stft_db_masked) * np.sign(sig_stft)) + (
        1j * sig_imag_masked
    )
#     if verbose:
#         print("Mask application:", td(seconds=time.time() - start))
#         start = time.time()
    # recover the signal
    recovered_signal = _istft(sig_stft_amp)
    recovered_spec = _amp_to_db(
        np.abs(_stft(recovered_signal)))

    return recovered_signal #audio data as if loaded from librosa.load
# return sig_stft_amp 




#recording parameters
CHUNK = 1024 #1024
FORMAT = pyaudio.paInt16
CHANNELS = pyaudio.PyAudio().get_default_input_device_info()['maxInputChannels'] #2
RATE = 16000 # 44100
EXTRA_SECONDS = 2.0
RECORD_SECONDS = NUM_NEW_CLIPS * MIN_CLIP_DURATION + EXTRA_SECONDS
BACKGROUND_RECORD_SECONDS = 3

def record_and_denoise( enroll = False):    
    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                    input=True, frames_per_buffer=CHUNK)


    
    
    
    LONG_STRING = "  \"She had your dark suit in greasy wash water all year. Don't ask me to carry an oily rag like that!\""

    print(" Recording {} seconds".format(RECORD_SECONDS - EXTRA_SECONDS))
    print("\n Speak the following sentence for recording: \n {}\n".format(LONG_STRING))

    print(' or\n')
    print(' You can speak your own secret phrases.')
    if enroll:
        print(' If you do so, please let us know your secret phrases:)\n\n')
    else: 
        print('\n')
    print(" Recording starts in soon...\n")
    
    frames_bg = []
    for i in range(0, int(RATE / CHUNK * (BACKGROUND_RECORD_SECONDS+1) ) ):
        data = stream.read(CHUNK, exception_on_overflow = False)
        frames_bg.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()

    
    
    if enroll:input(' Ready to start? (press enter)')
    else: pass#time.sleep(1)
    
    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                    input=True, frames_per_buffer=CHUNK)



    print(" Recording starts in 3 second...")
    time.sleep(2)   # start 1 second earlier
    frames = []
    
    print(" Speak now!")

    for i in tqdm(range(0, int(RATE / CHUNK * RECORD_SECONDS))):
        data = stream.read(CHUNK, exception_on_overflow = False)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()
    print(" Recording complete.")

#     audio_buffer = b''.join(frames)
#     bg_buffer = b''.join(frames_bg)
    audio_data = (np.frombuffer(b''.join(frames), dtype=np.int16)/32767)
    bg_data = (np.frombuffer(b''.join(frames_bg), dtype=np.int16)/32767)
    
    denoised_data = removeNoise(audio_data, bg_data).astype('float32')
    
    
    return denoised_data#, audio_buffer, bg_buffer


def write_recording(fpath, audio_data):  
    librosa.output.write_wav(fpath+'.wav', audio_data, sr=RATE)
    
    
#     wf = wave.open(fpath, 'wb') 
#     wf.setnchannels(CHANNELS)
#     wf.setsampwidth(2) #p.get_sample_size(FORMAT)
#     wf.setframerate(RATE)
#     wf.writeframes(buffer)#b''.join(frames)
#     wf.close()
    
def fpath_numbering(fpath, extension = '.wav'):
    while os.path.exists(fpath+extension):
        if fpath[-1].isalpha():
            fpath = fpath+'2'
        else:
            fpath = fpath[:-1]+str(int(fpath[-1])+1)
    return fpath
    

    
    
