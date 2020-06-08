import os
from utils import get_rel_path, TEST_PATH
# TRAIN_PATH = 'wav_train_subset'

train_user_ids = os.listdir(get_rel_path(TEST_PATH))
train_user_ids.sort()
train_user_ids = train_user_ids[1:] #delete .DS_Store
assert len(train_user_ids) > 0
with open(get_rel_path('test_clips_list.txt'), 'w') as f:
    f.write(train_user_ids[0]+' ')
    user = train_user_ids[0].split('_')[0]
    prev = user
    for i, ids in enumerate(train_user_ids[1:]):
        user = ids.split('_')[0]
        if user == prev:
            f.write(ids+' ')
        else:
            f.write('\n')
            f.write(ids+' ')
        prev = user
