import os
from tqdm import tqdm
import argparse
from utils import get_rel_path, TRAIN_PATH, TEST_PATH, find_username, CLIPS_LIST_FILE, TEST_CLIPS_LIST_FILE
# TRAIN_PATH = 'wav_train_subset'

parser = argparse.ArgumentParser()
parser.add_argument('--test',
                        default=False, action="store_true",
                        help="Preprocessing for test data")
parser.add_argument('--file_list',
                        default='', 
                        help="Provide the txt file of a list of audio paths if audio files are in subdirectories")


args = parser.parse_args()



if args.test:
    TRAIN_PATH, CLIPS_LIST_FILE = TEST_PATH, TEST_CLIPS_LIST_FILE

CLIP_PATH = get_rel_path(os.path.join(TRAIN_PATH,'../',CLIPS_LIST_FILE))
    
if args.file_list:
    with open(CLIP_PATH, 'w') as clip:
        with open(get_rel_path(args.file_list), 'r') as f:
            prev = None
            for line in f:
                user = find_username(line)
                if prev == user or prev is None:
                    clip.write(line.split('\n')[0]+' ')
                else:
                    clip.write('\n')
                    clip.write(line.split('\n')[0]+' ')
                prev = user
    
else:
    train_user_ids = os.listdir(get_rel_path(TRAIN_PATH))
    train_user_ids.sort()
    if train_user_ids[0] == '.DS_Store':
        train_user_ids = train_user_ids[1:] #delete .DS_Store
    assert len(train_user_ids) > 0
    with open(CLIP_PATH, 'w') as f:
        f.write(train_user_ids[0]+' ')
        user = find_username(train_user_ids[0])
        prev = user
        for i, ids in tqdm(enumerate(train_user_ids[1:])):
            user = ids.split('_')[0]
            if user == prev:
                f.write(ids+' ')
            else:
                f.write('\n')
                f.write(ids+' ')
            prev = user
