# vggvox

* Python adaptation of VGGVox speaker identification model, based on Nagrani et al 2017, "[VoxCeleb: a large-scale speaker identification dataset](https://arxiv.org/pdf/1706.08612.pdf)"

# Additional Data
Free ST American English Corpus

Source: http://www.openslr.org/resources/45/ST-AEDS-20180100_1-OS.tgz

Extract into /wav_train_subset

Move some subset of data of test users into /wav_test_subset

## Instructions
* Install python3 and the required packages
* To run:


### VBBA.py (Voice-Based Biometric Authentication)
```
python3 VBBA.py (optional argument)
```
```
(optional arguments):
  -h, --help            show this help message and exit
  -l, --list-current-users
                        Show current enrolled users
  -e, --enroll          Enroll a new user
  -v, --verify          Verify a user from the ones in the database
  -i, --identify        Identify a user
  -d, --delete          Delete user from database
  -c, --clear           Clear Database
  -u, --username        USERNAME
  -f, --with-file       Provide a recording file rather than record
```
For example:
```
python3 VBBA.py -e -u John_Doe   # Recording starts shortly. Enroll for John Doe.
```
```
python3 VBBA.py -e -u John_Doe -f recording/John_Doe_1.mp3  # Enroll for John Doe with recording file.
```
```
python3 VBBA.py -v -u John_Doe   # Recording starts shortly. Verify it with John_Doe's enrolled voice.
```
```
python3 VBBA.py -v -u John_Doe -f recordings/John_Doe_1.mp3   # Verify recording file with John_Doe's enrolled voice.
```
```
python3 VBBA.py -i   # Recording starts shortly. Detect and identify the voice among enrolled users.
```
```
python3 VBBA.py -i -f recordings/John_Doe_1.mp3   # Detect and identify the recording file among enrolled users.
```
```
python3 VBBA.py -l   # Display enrolled users.
```


### Train
```
python3 make_clips_list.py   # makes a clips_list.txt for training
```
```
python3 data_preprocessing.py   # makes np vectors from training audio files
```
```
python3 train.py train --test_users [users for test (default:None)] --n_epoches [number of epochs (default:1)]
```
For example:
```
python3 train.py train --test_users f0001 m0001 --n_epoches 15
```

### Test
```
python3 make_test_clips_list.py   # makes a test_clips_list.txt for testing
```
```
python3 test_data_prepocessing.py   # makes np vectors from testing audio files
```
```
python3 test.py test --threshold [threshold for cosine similarity for verification (default:0.95)]
```

### Recording
```
python3 recorder.py recorder --username [username]
```
Will save '(username).wav' in ./recordings/ folder with a stft file in ./recordings/stft
