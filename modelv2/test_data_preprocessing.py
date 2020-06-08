#!/usr/bin/env python
# coding: utf-8

# In[1]:


from utils import *


# In[3]:


def get_clip_duration(fname, subdirectory = None):
    if subdirectory:
        fname = ''.join(subdirectory.split('/'))+'/'+fname
    with contextlib.closing(wave.open(fname,'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
    return duration

    y, sr = librosa.load(fname, sr=None, offset=0., duration=MIN_CLIP_DURATION)
    return librosa.core.get_duration(y, sr=sr)


# In[4]:


import warnings
warnings.filterwarnings("ignore")



def get_user_clips(clips_per_user=CLIPS_PER_USER):
    ### all_user_clips is list of clips representing user and their list of clips
    all_user_clips = []
    
    with open(get_rel_path('test_clips_list.txt'), 'r') as f:
        for line in tqdm(f):
            paths = line.split()
#             paths = [get_rel_path("/".join(p.split("/"))) 
#                      for p in paths]
            paths = [p for p in paths if get_clip_duration(p,TEST_PATH) > MIN_CLIP_DURATION]
            if len(paths) < CLIPS_PER_USER:
                continue
                
            assert len(paths) > CLIPS_PER_USER 
            all_user_clips.extend(paths[: clips_per_user])
        
#     assert len(Counter(all_user_clips)) > 1
    return all_user_clips


# In[13]:


all_user_clips = get_user_clips()
print(len(all_user_clips), "clips")


# In[14]:


all_durations = np.array([get_clip_duration(p, TEST_PATH) for p in all_user_clips])
assert all_durations.min() > MIN_CLIP_DURATION


# In[15]:


def get_waveform(clip_list, offset=0., duration=MIN_CLIP_DURATION, subdirectory = None):
    all_x = []
    all_sr = []
    for path in tqdm(clip_list):
        if subdirectory:
            path = ''.join(subdirectory.split('/'))+'/'+path
        x, sr = librosa.load(path, sr=None, offset=offset, 
                             duration=duration)
        all_x.append(x)
        all_sr.append(sr)
        
    assert len(np.unique(np.array(all_sr))) == 1
    return all_x, all_sr


# In[16]:


all_x, all_sr = get_waveform(all_user_clips, subdirectory=TEST_PATH)


# In[17]:


assert len(np.unique(np.array([x.shape[0] for x in all_x]))) == 1, str(len(np.unique(np.array([x.shape[0] for x in all_x]))))


# In[18]:


def get_stft(all_x, nperseg=400, noverlap=239, nfft=1023):
    all_stft = []
    for x in tqdm(all_x):
        _, _, Z = scipy.signal.stft(x, window="hamming", 
                                       nperseg=nperseg,
                                       noverlap=noverlap,
                                       nfft=nfft)
        Z = sklearn.preprocessing.normalize(np.abs(Z), axis=1)        
        assert Z.shape[0] == 512
        all_stft.append(Z)
    
    return np.array(all_stft)


# In[19]:


all_stft = get_stft(all_x)
print(all_stft[0].shape)


# In[20]:


# librosa.display.specshow(all_stft[0], sr=all_sr[0])
# plt.show()


# In[21]:


# librosa.display.specshow(all_stft[-1], sr=all_sr[0])
# plt.show()


# In[22]:


def save_stft(all_stft, all_user_clips):
    all_stft_paths = []
    for i, user_path in tqdm(enumerate(all_user_clips)):
        user_stft = all_stft[i]
        stft_fname = '_'.join(user_path.split('/')[-3:])[:-4] + '.npy'
        stft_path = get_rel_path(os.path.join(TEST_STFT_FOLDER, stft_fname))
        np.save(stft_path, user_stft)
        all_stft_paths.append(stft_path)
    return all_stft_paths


# In[23]:
if not os.path.exists(TEST_STFT_FOLDER):
    os.mkdir(TEST_STFT_FOLDER)

stft_paths = save_stft(all_stft, all_user_clips)


# In[25]:


data = list(itertools.product(stft_paths, stft_paths))
df = pd.DataFrame(data, columns=["path1", "path2"])
df = df[~(df.path1 == df.path2)]  # to drop samples when path1 and path2 are same


# In[33]:




# In[39]:


df['user1'] = df['path1'].apply(lambda x: x[x.find('/')+1: x.find('_',x.find('/'))])
df['user2'] = df['path2'].apply(lambda x: x[x.find('/')+1: x.find('_',x.find('/'))])
df['label'] = (df.user1 == df.user2).astype('int')
df['label'] = np.abs(df.label - 1)
print("Total unique users", df.user1.nunique()) 
# assert df.user1.nunique() == TOTAL_USERS
# assert df.user2.nunique() == TOTAL_USERS
print("len", len(df))
df.sample(5)


# In[40]:


df.label.value_counts()




# In[22]:


# print(len(pairs_df))
# pairs_df.head()


# In[42]:


pairs_df = df


# In[43]:


# PAIRS_FILE = 'pairs_test.csv'
pairs_df.to_csv(TEST_PAIRS_FILE, index=False)


# In[ ]:





# In[ ]:




