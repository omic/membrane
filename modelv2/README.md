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

### Train
```
python3 make_clips_list.py # makes a clips_list.txt for training
```
```
python3 data_preprocessing.py # makes np vectors from training audio files
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
python3 make_test_clips_list.py # makes a test_clips_list.txt for testing
```
```
python3 test_data_prepocessing.py  # makes np vectors from testing audio files
```
```
python3 test.py test --threshold [threshold for cosine similarity for verification (default:0.95)]
```

### Recording
```
python3 recorder.py recorder --username [username]
```
Will save '(username).wav' in ./recordings/ folder with a stft file in ./recordings/stft
