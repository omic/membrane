#!/usr/bin/env python
# coding: utf-8


from utils import *
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--test',
                        default=False, action="store_true",
                        help="Preprocessing for test data")
args = parser.parse_args()

if args.test:
    TRAIN_PATH, PAIRS_FILE, STFT_FOLDER = TEST_PATH, TEST_PAIRS_FILE, TEST_STFT_FOLDER

    
CLIP_PATH = get_rel_path(os.path.join(TRAIN_PATH,'../',CLIPS_LIST_FILE))

def get_clip_duration(fname, subdirectory = None):
    if subdirectory:
        fname = ''.join(subdirectory.split('/'))+'/'+fname
#     with contextlib.closing(wave.open(fname,'r')) as f:
#         frames = f.getnframes()
#         rate = f.getframerate()
#         duration = frames / float(rate)
#     return duration

#     y, sr = librosa.load(fname, sr=RATE, offset=0., duration=MIN_CLIP_DURATION)
    return librosa.core.get_duration(filename = fname)

import warnings
warnings.filterwarnings("ignore")



def get_user_clips(clip_path = CLIP_PATH, clips_per_user=CLIPS_PER_USER):
    ### all_user_clips is list of clips representing user and their list of clips
    all_user_clips = []
    
    with open(get_rel_path(clip_path, 'r')) as f:
        num_users = 0
        pass_users = 0
        for line in tqdm(f):
            while pass_users<PASS_FIRST_USERS:
                pass_users+=1
                continue
            paths = line.split()
#             paths = [get_rel_path("/".join(p.split("/"))) 
#                      for p in paths]
#             paths = [p for p in paths if get_clip_duration(p) > MIN_CLIP_DURATION]#get_clip_duration(p, TRAIN_PATH)
            collected_paths = []
            i = 0
            for p in paths:
                if get_clip_duration(p) > MIN_CLIP_DURATION:
                    collected_paths.append(p)
                    i+=1
                if i == clips_per_user: break
            if len(collected_paths) < clips_per_user:
                continue
                
            all_user_clips.extend(collected_paths)
            num_users +=1
            if num_users >= TOTAL_USERS: break
#     assert len(Counter(all_user_clips)) > 1
    return all_user_clips


all_user_clips = get_user_clips()
print(len(all_user_clips), "clips")


# all_durations = np.array([get_clip_duration(p, TRAIN_PATH) for p in all_user_clips])
# assert all_durations.min() > MIN_CLIP_DURATION


###########Augmentation###########
def time_shift(data, sampling_rate = 16000, shift_max = 1, shift_direction = 'both'):
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
              
              
# from https://stackoverflow.com/questions/33933842/how-to-generate-noise-in-frequency-range-with-numpy
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
          
              
with open(get_rel_path(BACKGROUND_LIST_PATH), 'r') as f:
    bg_array = []
    for line in f:
        bg_array.append(np.load(line.split('\n')[0]))             
              
SHIFT_CHANCE = 0.5 #20% chance of shifting
W_NOISE_CHANCE = 0.8 #80% chance of white noise
NOISE_CHANCE = 0.5 #50% chance of putting noise
def augmentation(src_data, src_rate = RATE):
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
              
              
              
              
def get_waveform(clip_list, offset=0., duration=MIN_CLIP_DURATION, subdirectory = None):
    all_x = []
    all_sr = []
    for path in tqdm(clip_list):
        if subdirectory:
            path = ''.join(subdirectory.split('/'))+'/'+path
        x, _ = librosa.load(path, offset=offset, 
                             duration=duration, sr = RATE)
        if AUGMENT:
            x, noise = augmentation(x)
            x = removeNoise(x, noise)
        all_x.append(x)
        all_sr.append(sr)
        
#     assert len(np.unique(np.array(all_sr))) == 1
    return all_x, all_sr


all_x, all_sr = get_waveform(all_user_clips)# subdirectory=TRAIN_PATH



assert len(np.unique(np.array([x.shape[0] for x in all_x]))) == 1, str(len(np.unique(np.array([x.shape[0] for x in all_x]))))


## in utils.py
# def get_stft(all_x, nperseg=400, noverlap=239, nfft=1023):
#     all_stft = []
#     for x in tqdm(all_x):
#         _, _, Z = scipy.signal.stft(x, window="hamming", 
#                                        nperseg=nperseg,
#                                        noverlap=noverlap,
#                                        nfft=nfft)
#         Z = sklearn.preprocessing.normalize(np.abs(Z), axis=1)        
#         assert Z.shape[0] == 512
#         all_stft.append(Z)
    
#     return np.array(all_stft)


all_stft = get_stft(all_x)
print('stft shape: ',all_stft[0].shape)


# In[20]:


# librosa.display.specshow(all_stft[0], sr=all_sr[0])
# plt.show()


# In[21]:


# librosa.display.specshow(all_stft[-1], sr=all_sr[0])
# plt.show()



def save_stft(all_stft, all_user_clips):
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


# In[25]:


data = list(itertools.product(stft_paths, stft_paths))
stft_len = len(stft_paths)
print('stft_len: ',stft_len)
# if TRAIN_PAIR_SAMPLES
# data = zip(np.random.choice(stft_paths, size = TRAIN_PAIR_SAMPLES),np.random.choice(stft_paths, size = TRAIN_PAIR_SAMPLES))
df = pd.DataFrame(data, columns=["path1", "path2"])
df = df[~(df.path1 == df.path2)]  # to drop samples when path1 and path2 are same


# In[33]:



# In[39]:


# df['user1'] = df['path1'].apply(lambda x: x[x.find('/')+1: x.find('_',x.find('/'))])
# df['user2'] = df['path2'].apply(lambda x: x[x.find('/')+1: x.find('_',x.find('/'))])


df['user1'] = df['path1'].apply(find_username)
df['user2'] = df['path2'].apply(find_username)

df['label'] = (df.user1 == df.user2).astype('int')
df['label'] = np.abs(df.label - 1)
print("Total unique users", df.user1.nunique()) 
# assert df.user1.nunique() == TOTAL_USERS
# assert df.user2.nunique() == TOTAL_USERS
print("len", len(df))
print(df.sample(5))


# In[40]:


print(df.label.value_counts())




# In[22]:


# print(len(pairs_df))
# pairs_df.head()


# In[42]:


pairs_df = df #.sample(TRAIN_PAIR_SAMPLES)


# In[43]:

PAIRS_FILE_PATH = os.path.join(TRAIN_PATH, '../', PAIRS_FILE)

# PAIRS_FILE = 'pairs_test.csv'
pairs_df.to_csv(PAIRS_FILE_PATH, index=False)


# In[ ]:





# In[ ]:




