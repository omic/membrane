#! /usr/bin/env python3
from utils import *


# def record(fpath):
#     CHUNK = 2048 #1024
#     FORMAT = pyaudio.paInt16
#     CHANNELS = 2
#     RATE = 16000 # 44100
#     EXTRA_SECONDS = 2.0
#     RECORD_SECONDS = NUM_NEW_CLIPS * MIN_CLIP_DURATION + EXTRA_SECONDS

#     LONG_STRING = "She had your dark suit in greasy wash water all year. Don't ask me to carry an oily rag like that!"

#     print("Recording {} seconds".format(RECORD_SECONDS - EXTRA_SECONDS))
#     print("\n Speak the following sentence for recording: \n {} \n".format(LONG_STRING))

#     p = pyaudio.PyAudio()

#     stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE,
#                     input=True, frames_per_buffer=CHUNK)

#     time.sleep(1)

#     print("Recording starts in 3 seconds...")
#     time.sleep(2)   # start 1 second earlier
#     print("Speak now!")
#     frames = []

#     for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
#         data = stream.read(CHUNK, exception_on_overflow = False)
#         frames.append(data)

#     stream.stop_stream()
#     stream.close()
#     p.terminate()

#     print("Recording complete")

#     wf = wave.open(fpath, 'wb') 
#     wf.setnchannels(CHANNELS)
#     wf.setsampwidth(p.get_sample_size(FORMAT))
#     wf.setframerate(RATE)
#     wf.writeframes(b''.join(frames))
#     wf.close()



# def split_recording(recording=ENROLL_RECORDING_FNAME):
#     wav, sr = librosa.load(recording)
#     RECORD_SECONDS = int(NUM_NEW_CLIPS * MIN_CLIP_DURATION)
#     all_x = []
#     for offset in range(0, RECORD_SECONDS, int(MIN_CLIP_DURATION)):
#         x, sr = librosa.load(recording, sr=16000, offset=offset,
#                              duration=MIN_CLIP_DURATION)

#         all_x.append(x)

#     return get_stft(all_x)

# def save_stft(all_stft, recording=ENROLL_RECORDING_FNAME):
#     all_stft_paths = []
#     for i in tqdm(range(len(all_stft))):
#         user_stft = all_stft[i]
#         stft_fname = '_'.join(recording.split('/')[-3:])[:-4] + '.npy'
#         stft_path = get_rel_path(os.path.join(RECORDING_STFT_FOLDER, stft_fname))
#         np.save(stft_path, user_stft)
#         all_stft_paths.append(stft_path)
#     return all_stft_paths


# In[23]:


def recorder(opt):
    ENROLL_RECORDING_FNAME = opt.username+'.wav'
    if not os.path.exists(RECORDING_PATH):
        os.mkdir(RECORDING_PATH)
    if not os.path.exists(RECORDING_STFT_FOLDER):
        os.mkdir(RECORDING_STFT_FOLDER)
    fpath = os.path.join(RECORDING_PATH, ENROLL_RECORDING_FNAME)
    #''.join(RECORDING_PATH.split('/'))+'/'+ENROLL_RECORDING_FNAME
    record(fpath)
    stfts = split_recording(fpath)
    print(f'stfts lengths: {len(stfts)}')
    print(f'stft shape: {stfts[0].shape}')
    stft_paths = save_stft(stfts, ENROLL_RECORDING_FNAME)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	subparsers = parser.add_subparsers()
	parser_recorder = subparsers.add_parser('recorder')
# 	parser_test.add_argument('--test_users', nargs='*', default = None)#, default = 'data/wav/enroll/19-enroll.wav')
	parser_recorder.add_argument('--username', default = ENROLL_RECORDING_FNAME)#, default = 'data/wav/test/19-test.wav')
# 	parser_scoring.add_argument('--metric', default = 'cosine')
# 	parser_scoring.add_argument('--threshold', default = 0.1)
	parser_recorder.set_defaults(func=recorder)
	opt = parser.parse_args()
	opt.func(opt)