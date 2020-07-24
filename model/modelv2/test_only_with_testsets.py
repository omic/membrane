#!/usr/bin/env python
# coding: utf-8

from utils import *
from network import *

def test(opt):
    TEST_PAIRS_FILE = opt.pairs
    if opt.val:
        PAIRS_FILE = opt.val
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)


    weights = load_pretrained_weights()

    checkpoints = os.listdir(get_rel_path('checkpoints/'))
#     print("\n".join(checkpoints))
    checkpoints.sort()
    print(checkpoints[-1])

    test_model ,_, _ = load_saved_model(checkpoints[-1], test=True)


    class VoxCelebTestDataset(Dataset):

        def __init__(self, pairs_fname,  users=None, n_users=5, clips_per_user=5):
            if users:
                pairs_file = pd.read_csv(get_rel_path(pairs_fname))
                user_subset = pairs_file[pairs_file.user1.isin(users)]
                self.users =  user_subset.user1.unique()
                self.spec = user_subset[user_subset.user1.isin(self.users)]
                self.spec = self.spec.drop_duplicates(subset = ['path1'])[['user1', 'path1']].values
            else:
                pairs_file = pd.read_csv(get_rel_path(pairs_fname))
                user_subset = pairs_file
                self.users =  pairs_file.user1.unique()
                self.spec = user_subset[user_subset.user1.isin(self.users)]
                self.spec = self.spec.drop_duplicates(subset = ['path1'])[['user1', 'path1']].values

        def __len__(self):
            return len(self.spec)

        def __getitem__(self, idx):
            spec1_path = get_rel_path(self.spec[idx][1])
            user_id = self.spec[idx][0]
            spec1 = np.load(spec1_path)
            sample = {'spec': spec1, 'user_id': user_id}

            return sample

    test_batch_size = 1

#     voxceleb_train_dataset = VoxCelebTestDataset(PAIRS_FILE, training_users, clips_per_user=NUM_NEW_CLIPS, n_users=20)
    voxceleb_total_dataset = VoxCelebTestDataset(PAIRS_FILE, clips_per_user=NUM_NEW_CLIPS, n_users=20) #, total_users
    voxceleb_test_dataset = VoxCelebTestDataset(TEST_PAIRS_FILE, clips_per_user=NUM_NEW_CLIPS, n_users=20)#, test_users

    def get_user_model(dataset):
        user_dict = OrderedDict()

        for i, data in enumerate(dataset):
            spec, user_id = data['spec'], data['user_id']
            spec = torch.tensor(spec)
            spec = spec.view(test_batch_size, 1, spec.shape[0], spec.shape[1])
            spec = spec.to(device)
            out = test_model.forward_single(spec)
            out = out.view(out.shape[0], out.shape[1])

            if user_dict.get(user_id, None) is not None:
                  user_dict[user_id].append(out.detach().cpu().numpy())

            else:
                  user_dict[user_id] = [out.detach().cpu().numpy()]
        mean_dict = {}
        for user_id, emb_list  in user_dict.items():
            emb_list = np.array(emb_list)
            mean_emb = np.mean(emb_list, axis = 0)
            mean_dict[user_id] = mean_emb

        return user_dict, mean_dict

    total_user_embeddings ,total_mean_dict = get_user_model(voxceleb_total_dataset)

    test_user_embeddings ,test_mean_dict = get_user_model(voxceleb_total_dataset)

    threshold = opt.threshold
    correct, incorrect, FP, FN = 0, 0, 0, 0
    user_label = {}
    k=0
    mean_data = []
    for user in total_mean_dict:
        user_label[user]=k
        mean_data.append(total_mean_dict[user])
        k+=1
    mean_data = np.vstack(mean_data)
    distances_to_truth = {}
    for user in test_user_embeddings:
        user_emb_i = np.vstack(test_user_embeddings[user])
        users_cosine_similarity = cosine_similarity(user_emb_i, mean_data)
        distances_to_truth[user] = users_cosine_similarity[:,user_label[user]]
        pred = np.argmax(users_cosine_similarity, axis=1)
        correct_i = (pred==user_label[user]).sum()
        correct += correct_i
        incorrect += (user_emb_i.shape[0] - correct_i)

    acc = correct / (correct + incorrect)
    GT_distances = np.vstack(list(distances_to_truth.values())).flatten()
    veri = np.sum(GT_distances>threshold)/len(GT_distances)#GroundTruth

    print(f'num of correct: {correct}')
    print(f'num of incorrect: {incorrect}')
    print(f'accuracy for argmax identification: {acc*100:.2f}%')
    print(f'accuracy for verification with threshold={threshold}: {veri*100:.2f}%')


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	subparsers = parser.add_subparsers()
	parser_test = subparsers.add_parser('test')
# 	parser_test.add_argument('--test_users', nargs='*', default = None)
	parser_test.add_argument('--threshold', default = 0.99)
	parser_test.add_argument('--pairs', default = os.path.join(TEST_PATH,'../',TEST_PAIRS_FILE))
	parser_test.add_argument('--val_pairs',dest = 'val', default = '')
	parser_test.set_defaults(func=test)
	opt = parser.parse_args()
	opt.func(opt)



