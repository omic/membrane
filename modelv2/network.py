from utils import *


class VggVox(nn.Module):
    '''
    Class for CNN architecture (VGGvox)
    '''
    def __init__(self, weights=None):
        super(VggVox, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=1,
                                             out_channels=n_f1,
                                             kernel_size=conv_kernel1,
                                             stride=s1,
                                             padding=p1),
                                    nn.BatchNorm2d(num_features=n_f1),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=pool_kernel1,
                                                 stride=pool_s1))

        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=n_f1,
                                         out_channels=n_f2,
                                         kernel_size=conv_kernel2,
                                         stride=s2,
                                         padding=p2),
                                nn.BatchNorm2d(num_features=n_f2),
                                nn.ReLU(),
                                nn.MaxPool2d(kernel_size=pool_kernel2,
                                             stride=pool_s2))


        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=n_f2,
                                             out_channels=n_f3,
                                             kernel_size=conv_kernel3,
                                             stride=s3,
                                             padding=p3),
                                    nn.BatchNorm2d(num_features=n_f3),
                                    nn.ReLU())

        self.conv4 = nn.Sequential(nn.Conv2d(in_channels=n_f3,
                                             out_channels=n_f4,
                                             kernel_size=conv_kernel4,
                                             stride=s4,
                                             padding=p4),
                                   nn.BatchNorm2d(num_features=n_f4),
                                   nn.ReLU())

        self.conv5 = nn.Sequential(nn.Conv2d(in_channels=n_f4,
                                             out_channels=n_f5,
                                             kernel_size=conv_kernel5,
                                             stride=s5,
                                             padding=p5),
                                   nn.BatchNorm2d(num_features=n_f5),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=(pool_kernel5_x, pool_kernel5_y),
                                                stride=(pool_s5_x, pool_s5_y)))

        self.fc6 = nn.Sequential(nn.Conv2d(in_channels=n_f5,
                                             out_channels=n_f6,
                                             kernel_size=(conv_kernel6_x, conv_kernel6_y),
                                             stride=s6),
                                 nn.BatchNorm2d(num_features=n_f6),
                                 nn.ReLU())

        self.global_pool = nn.AvgPool2d

        self.fc7 = nn.Sequential(nn.Conv2d(in_channels=n_f6,
                                           out_channels=n_f7,
                                           kernel_size=conv_kernel7,
                                           stride=s7),
                                 nn.ReLU())

        self.fc8 = nn.Sequential(nn.Conv2d(in_channels=n_f7,
                                           out_channels=n_f8,
                                           kernel_size=conv_kernel8,
                                           stride=s8))

        if weights is not None:
            self.conv1[0].weight = torch.nn.Parameter(torch.from_numpy(weights['conv1'][0]))
            self.conv1[1].weight = torch.nn.Parameter(torch.from_numpy(weights['conv1'][1]))

            self.conv2[0].weight = torch.nn.Parameter(torch.from_numpy(weights['conv2'][0]))
            self.conv2[1].weight = torch.nn.Parameter(torch.from_numpy(weights['conv2'][1]))

            self.conv3[0].weight = torch.nn.Parameter(torch.from_numpy(weights['conv3'][0]))
            self.conv3[1].weight = torch.nn.Parameter(torch.from_numpy(weights['conv3'][1]))

            self.conv4[0].weight = torch.nn.Parameter(torch.from_numpy(weights['conv4'][0]))
            self.conv4[1].weight = torch.nn.Parameter(torch.from_numpy(weights['conv4'][1]))

            self.conv5[0].weight = torch.nn.Parameter(torch.from_numpy(weights['conv5'][0]))
            self.conv5[1].weight = torch.nn.Parameter(torch.from_numpy(weights['conv5'][1]))

            self.fc6[0].weight = torch.nn.Parameter(torch.from_numpy(weights['fc6'][0]))
            self.fc6[1].weight = torch.nn.Parameter(torch.from_numpy(weights['fc6'][1]))

            self.fc7[0].weight = torch.nn.Parameter(torch.from_numpy(weights['fc7'][0]))

    def forward_single(self, x):
        x = self.conv1(x)
#         print(x.shape) #let's check the shape
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
#         print('after conv5:',x.shape) #let's check the shape
        x = self.fc6(x)
#         print('after fc6:',x.shape) #let's check the shape
        x = self.global_pool(kernel_size=x.size()[2:])(x)
#         print('after avg pool:',x.shape) #let's check the shape
        x = self.fc7(x)
        out = self.fc8(x)
        out = out.view(-1, out.shape[1])
        return out

    def forward(self, input1, input2):
        output1 = self.forward_single(input1)
        output2 = self.forward_single(input2)
        return output1, output2


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive


class VoxCelebDataset(Dataset):
    def __init__(self, pairs_fname=PAIRS_FILE, n_users=TRAINING_USERS, clips_per_user=CLIPS_PER_USER, train=True, test_users = None):
        pairs_file = pd.read_csv(get_rel_path(pairs_fname))
        self.all_user_ids = sorted(pairs_file.user1.unique())
        if test_users:
            self.training_users = [user for user in self.all_user_ids if user not in test_users]
        else:
            self.training_users = self.all_user_ids[: n_users]
        
        
        def balance_data(df):
            pairs_df = []
            for user in df.user1.unique():
                user_df = df[df.user1 == user]
                similar = user_df[user_df['label'] == 0].sample(n=SIMILAR_PAIRS)
                dissimilar = user_df[user_df['label'] == 1].sample(n=DISSIMILAR_PAIRS)
                pairs_df.append(pd.concat([similar, dissimilar]))

            pairs_df = pd.concat(pairs_df)
            return pairs_df
        
        if train:
            user1_subset = pairs_file[pairs_file.user1.isin(self.training_users)]
            user2_subset = user1_subset[user1_subset.user2.isin(self.training_users)]
        else:
            user1_subset = pairs_file[~pairs_file.user1.isin(self.training_users)]
            user2_subset = user1_subset[~user1_subset.user2.isin(self.training_users)]            

        pairs_df = balance_data(user2_subset)
        
        if train:
            assert len(pairs_df[pairs_df.user1.isin(self.training_users)]) == len(pairs_df)
            assert len(pairs_df[pairs_df.user2.isin(self.training_users)]) == len(pairs_df)
        else:
            assert len(pairs_df[~pairs_df.user1.isin(self.training_users)]) == len(pairs_df)
            assert len(pairs_df[~pairs_df.user2.isin(self.training_users)]) == len(pairs_df)

        self.spec = pairs_df[['path1', 'path2', 'label']].values

    def __len__(self):
        return len(self.spec)

    def __getitem__(self, idx):
        spec1_path = get_rel_path(self.spec[idx][0])
        spec2_path = get_rel_path(self.spec[idx][1])
        label = int(self.spec[idx][2])

        spec1 = np.load(spec1_path)
        spec2 = np.load(spec2_path)

        spec1 = np.expand_dims(spec1, axis=0)
        spec2 = np.expand_dims(spec2, axis=0)

        assert spec1.ndim == 3, spec2.ndim == 3

        sample = {'spec1': spec1, 'spec2': spec2, 'label': label}

        return sample


def load_saved_model(fname, test=True):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    new_model_dict = VggVox()
    checkpoint_path = get_rel_path(os.path.join(CHECKPOINTS_FOLDER, fname))
    checkpoint = torch.load(checkpoint_path, map_location=device)

    new_model_dict.load_state_dict(checkpoint['state_dict'])
    if test:
        model = new_model_dict.eval()
    print("Loading model in test mode:", test)
    model = model.to(device)

    new_optimizer = optim.Adam(params=model.parameters())
    new_optimizer.load_state_dict(checkpoint['optim_dict'])
    epoch = checkpoint['epoch']

    return model, new_model_dict, new_optimizer
