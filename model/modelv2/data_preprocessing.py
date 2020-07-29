#!/usr/bin/env python
# coding: utf-8
from utils import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--test',
                        default=False, action="store_true",
                        help="Preprocessing for test data")
parser.add_argument('--for-enroll',dest = 'val',type=int,
                        default=0,
                        help="Number of each users data taken out for enrollment, which will give separate pairs file.")
args = parser.parse_args()

if args.test:
    TRAIN_PATH, PAIRS_FILE, STFT_FOLDER, CLIPS_LIST_FILE, CLIPS_PER_USER = TEST_PATH, TEST_PAIRS_FILE, TEST_STFT_FOLDER, TEST_CLIPS_LIST_FILE, TEST_CLIPS_PER_USER
    PASS_FIRST_USERS = 0

CLIP_PATH = get_rel_path(os.path.join(TRAIN_PATH,'../',CLIPS_LIST_FILE))

def get_clip_duration(fname:str, subdirectory = None):
    """
    Get duration of the clip

    fname: path to an audio file
    subdirectory: pass subdirectory path if the file exists in subdirectory from the root directory.
    """
    if subdirectory:
        fname = ''.join(subdirectory.split('/'))+'/'+fname
    return librosa.core.get_duration(filename = fname)

import warnings
warnings.filterwarnings("ignore")

def get_user_clips(clip_path:str = CLIP_PATH, clips_per_user:int=CLIPS_PER_USER):
    """
    Make a list of 'clips_per_user' number of clips longer than 'MIN_CLIP_DURATION' of each user.

    clip_path: path to .txt file which has a list of paths to all audio clips.
    clips_per_user: the number of clips to make per user
    """
    ### all_user_clips is list of clips representing user and their list of clips
    all_user_clips = []
    with open(get_rel_path(clip_path, 'r')) as f:
        num_users = 0
        pass_users = 0
        for line in tqdm(f):
            if pass_users<PASS_FIRST_USERS:
                pass_users+=1
                continue

            paths = line.split()
            collected_paths = []
            i = 0
            for p in paths:
                if get_clip_duration(p) > MIN_CLIP_DURATION:
                    collected_paths.append(p)
                    i+=1
                if clips_per_user is not None:
                    if i >= clips_per_user: break
            if clips_per_user is not None:
                if len(collected_paths) < clips_per_user:
                    continue

            all_user_clips.extend(collected_paths)
            num_users +=1
            if num_users >= TOTAL_USERS: break
    return all_user_clips

all_user_clips = get_user_clips()
print(len(all_user_clips), "clips")

###########Augmentation###########
def time_shift(data, sampling_rate = 16000, shift_max = 1, shift_direction = 'both'):
    """
    data: audio data as a array
    sampling_rate: sampling rate of the audio data
    shift_max: maximum seconds to shift
    shift_direction: a direction to shift
    """
    shift = np.random.randint(sampling_rate * shift_max)
    if shift_direction == 'right':
        shift = -shift
    elif shift_direction == 'both':
        direction = np.random.randint(0, 2)
        if direction == 1:
            shift = -shift
    augmented_data = np.roll(data, shift)
    # Set to silence for heading/ tailing
    if shift > 0:
        augmented_data[:shift] = 0
    else:
        augmented_data[shift:] = 0
    return augmented_data

###generating noise
### from https://stackoverflow.com/questions/33933842/how-to-generate-noise-in-frequency-range-with-numpy
def fftnoise(f):
    f = np.array(f, dtype="complex")
    Np = (len(f) - 1) // 2
    phases = np.random.rand(Np) * 2 * np.pi
    phases = np.cos(phases) + 1j * np.sin(phases)
    f[1 : Np + 1] *= phases
    f[-1 : -1 - Np : -1] = np.conj(f[1 : Np + 1])
    return np.fft.ifft(f).real

def band_limited_noise(min_freq, max_freq, samples=1024, samplerate=1):
    freqs = np.abs(np.fft.fftfreq(samples, 1 / samplerate))
    f = np.zeros(samples)
    f[np.logical_and(freqs >= min_freq, freqs <= max_freq)] = 1
    return fftnoise(f)
###

with open(get_rel_path(BACKGROUND_LIST_PATH), 'r') as f:
    bg_array = []
    for line in f:
        bg_array.append(np.load(line.split('\n')[0]))

SHIFT_CHANCE = 0.5 #20% chance of shifting
W_NOISE_CHANCE = 0.8 #80% chance of white noise
NOISE_CHANCE = 0.5 #50% chance of putting noise

def augmentation(src_data, src_rate = RATE):
    """
    src_data: source audio data as in array.
    src_rate: sampling rate of the source audio.
    """
    if np.random.choice(2,p=[1-SHIFT_CHANCE,SHIFT_CHANCE]):
        src_data = time_shift(src_data)
    if np.random.choice(2,p=[1-W_NOISE_CHANCE,W_NOISE_CHANCE]):
        noise = band_limited_noise(min_freq=4000, max_freq = 12000, samples=len(src_data), samplerate=src_rate)*10
        snr = max(1,np.random.normal(5,2.5))
        noise = noise/snr # white noise
    else: noise = np.zeros(src_data.shape)
    if np.random.choice(2,p=[1-NOISE_CHANCE,NOISE_CHANCE]):
        bg_idx = np.random.randint(len(bg_array))
        offset = np.random.randint(len(bg_array[bg_idx])-len(src_data))
        bg_noise = bg_array[bg_idx][offset:offset+len(src_data)]
#     noise = noise[: len(src_data)]
        snr = max(1,np.random.normal(10,5)) # signal to noise ratio
        bg_noise = bg_noise/snr
        noise = noise+bg_noise
        src_data = src_data +noise
    return src_data, noise


##################################

def get_waveform(clip_list, offset=0., duration=MIN_CLIP_DURATION, subdirectory = None, augment:bool = AUGMENT):
    """
    clip_lists: a list of clips representing user and their list of clips.
    offset: offset when reading audio file.
    duration: duration to read audio file.
    subdirectory: pass subdirectory path if the file exists in subdirectory from the root directory.
    augment: whether to augment audio.
    """
    all_x = []
    all_sr = []
    for path in tqdm(clip_list):
        if subdirectory:
            path = ''.join(subdirectory.split('/'))+'/'+path
        x, _ = librosa.load(path, offset=offset,
                             duration=duration, sr = RATE)
        if augment:
            x, noise = augmentation(x)
            x = removeNoise(x, noise)
        all_x.append(x)
        all_sr.append(sr)
#     assert len(np.unique(np.array(all_sr))) == 1
    return all_x, all_sr

all_x, all_sr = get_waveform(all_user_clips)# subdirectory=TRAIN_PATH

assert len(np.unique(np.array([x.shape[0] for x in all_x]))) == 1, str(len(np.unique(np.array([x.shape[0] for x in all_x]))))

all_stft = get_stft(all_x)
print('stft shape: ',all_stft[0].shape)

def save_stft(all_stft, all_user_clips):
    """
    all_stft: array of STFTs for all clips.
    all_user_clips: a list of clips representing user and their list of clips.
    """
    all_stft_paths = []
    for i, user_path in tqdm(enumerate(all_user_clips)):
        user_stft = all_stft[i]
#         stft_fname = '_'.join(user_path.split('/')[-3:])[:-4] + '.npy'
        stft_fname = user_path.rsplit('/')[-1].split('.')[0] +'.npy'
        stft_path = get_rel_path(os.path.join(STFT_FOLDER, stft_fname))
        np.save(stft_path, user_stft)
        all_stft_paths.append(stft_path)
    return all_stft_paths

stft_paths = save_stft(all_stft, all_user_clips)

if args.val:
    val_df = pd.DataFrame()
    test_df = pd.DataFrame()
    df = pd.DataFrame(stft_paths, columns = ['path'])
    df['user'] = df['path'].apply(find_username)
    for user in df['user'].unique():
        user_df = df[df['user'] == user]
        val_df = pd.concat([val_df,user_df[:args.val]])
        test_df = pd.concat([test_df,user_df[args.val:]])
    test_data_path = list(itertools.permutations(test_df['path'], 2))
    test_data_user = list(itertools.permutations(test_df['user'],2))
    val_data_path = list(itertools.permutations(val_df['path'], 2))
    val_data_user = list(itertools.permutations(val_df['user'], 2))
    df = pd.concat([pd.DataFrame(test_data_path,columns = ['path1','path2']),\
    pd.DataFrame(test_data_user,columns = ['user1','user2'])],axis=1)
    val_df = pd.concat([pd.DataFrame(val_data_path,columns = ['path1','path2']),\
    pd.DataFrame(val_data_user,columns = ['user1','user2'])],axis=1)
else:
    data = list(itertools.permutations(stft_paths, 2))
    stft_len = len(stft_paths)
    print('stft_len: ',stft_len)
    # if TRAIN_PAIR_SAMPLES
    # data = zip(np.random.choice(stft_paths, size = TRAIN_PAIR_SAMPLES),np.random.choice(stft_paths, size = TRAIN_PAIR_SAMPLES))
    df = pd.DataFrame(data, columns=["path1", "path2"])

    df['user1'] = df['path1'].apply(find_username)
    df['user2'] = df['path2'].apply(find_username)

def make_label(df):
    df['label'] = (df.user1 == df.user2).astype('int8')
    df['label'] = np.abs(df.label - 1)
    return df

df = make_label(df)
if args.val: val_df = make_label(val_df)
print("Total unique users", df.user1.nunique())
# assert df.user1.nunique() == TOTAL_USERS
# assert df.user2.nunique() == TOTAL_USERS
print("len", len(df))
print(df.sample(5))
print(df.label.value_counts())

pairs_df = df #.sample(TRAIN_PAIR_SAMPLES)
PAIRS_FILE_PATH = os.path.join(TRAIN_PATH, '../', PAIRS_FILE)
# PAIRS_FILE = 'pairs_test.csv'
pairs_df.to_csv(PAIRS_FILE_PATH, index=False)
if args.val:
    pairs_df = val_df #.sample(TRAIN_PAIR_SAMPLES)
    PAIRS_FILE_PATH = os.path.join(TRAIN_PATH, '../','val_'+ PAIRS_FILE)
    # PAIRS_FILE = 'pairs_test.csv'
    pairs_df.to_csv(PAIRS_FILE_PATH, index=False)
