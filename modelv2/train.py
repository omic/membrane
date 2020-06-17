#!/usr/bin/env python
# coding: utf-8

# In[21]:


from utils import *
from network import *


# In[22]:

def train(n_epochs = N_EPOCHS,  pairs_file = PAIRS_FILE, test_users = None ):#opt
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    weights = load_pretrained_weights()

    model = VggVox(weights=weights)
    model = model.to(device)

    criterion = ContrastiveLoss()
    criterion = criterion.to(device)

    loss_list = []
    best_loss = torch.autograd.Variable(torch.tensor(np.inf)).float()

#     LEARNING_RATE = 1e-3
#     N_EPOCHS = 1 #15
#     N_EPOCHS = int(opt.n_epochs)
#     N_EPOCH = n_epochs
#     BATCH_SIZE = 64

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # model, _, optimizer = load_saved_model("checkpoint_20181211-030043_0.014894404448568821.pth.tar", test=False)

#     test_users = opt.test_users

    voxceleb_dataset = VoxCelebDataset(pairs_file, test_users = test_users)#PAIRS_FILE
    train_dataloader = DataLoader(voxceleb_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=4)
    n_batches = int(len(voxceleb_dataset) / BATCH_SIZE)

    print("training unique users", len(voxceleb_dataset.training_users))
    print("training samples", len(voxceleb_dataset))
    print("batches", int(len(voxceleb_dataset) / BATCH_SIZE))

    for epoch in range(1, N_EPOCHS+1):
        running_loss = torch.zeros(1)

        for i_batch, data in enumerate(train_dataloader, 1):
            mfcc1, mfcc2, label = data['spec1'], data['spec2'], data['label']
            mfcc1 = Variable(mfcc1.float(), requires_grad=True).to(device)
            mfcc2 = Variable(mfcc2.float(), requires_grad=True).to(device)
            label = Variable(label.float(), requires_grad=True).to(device)

            output1, output2 = model(mfcc1.float(), mfcc2.float())

            optimizer.zero_grad()

            loss = criterion(output1, output2, label.float())

    #         assert mfcc1.dim() == mfcc2.dim() == 4
    #         assert output1.dim() == output2.dim() == 2
    #         assert loss.requires_grad and output1.requires_grad and output2.requires_grad
    #         assert loss.grad_fn is not None and output1.grad_fn is not None and output2.grad_fn is not None

    #         print("loss", loss, loss.requires_grad, loss.grad_fn)
    #         print("output1", output1.shape, output1.requires_grad, output1.grad_fn, output1.device)
    #         print("output2", output2.shape, output2.requires_grad, output2.grad_fn, output2.device)

            loss.backward()

    #         assert mfcc1.requires_grad and mfcc2.requires_grad
    #         for name, param in model.named_parameters():
    #             assert param.requires_grad and param.grad is not None, (name, param.requires_grad, param.grad)

            optimizer.step()

            loss_list.append(loss.item())
            running_loss += loss.item()
            if i_batch % int(n_batches/min(20,n_batches)) == 0:
                print("Epoch {}/{}  Batch {}/{} \nCurrent Batch Loss {}\n".format(epoch, N_EPOCHS, i_batch, n_batches, loss.item()))
        epoch_loss = running_loss / len(voxceleb_dataset)
        print("==> Epoch {}/{} Epoch Loss {}".format(epoch, N_EPOCHS, epoch_loss.item()))

        is_best = epoch_loss < best_loss
        if is_best:
            best_loss = epoch_loss

            save_checkpoint({'epoch': epoch,
                             'state_dict': model.state_dict(),
                             'optim_dict': optimizer.state_dict()},
                            loss=epoch_loss)
        else:
            print("### Epoch Loss did not improve\n")

#     plt.plot(loss_list[50:])
#     plt.show()



if __name__ == '__main__':
# 	get_id_result()
    parser = argparse.ArgumentParser()
# 	subparsers = parser.add_subparsers()
# 	parser_train = subparsers.add_parser('train')
# 	parser_train.add_argument('--test_users', nargs='*', default = None)#, default = 'data/wav/enroll/19-enroll.wav')
# 	parser_train.add_argument('--n_epochs', default = 1)#, default = 'data/wav/test/19-test.wav')
# # 	parser_scoring.add_argument('--metric', default = 'cosine')
# # 	parser_scoring.add_argument('--threshold', default = 0.1)
# 	parser_train.set_defaults(func=train)
# 	opt = parser.parse_args()
# 	opt.func(opt)
#     parser.add_argument('--augment',
#                         default=False, action="store_true",
#                         help="Train with augmented data")

    parser.add_argument('--test-users', nargs='*', default = None,
                       help="Leave out test sets from train sets")#, default = 'data/wav/enroll/19-enroll.wav')
    parser.add_argument('--n-epochs', default = 1,
                       help = 'Set the number of epochs')
    parser.add_argument('--pairs-file', default = os.path.join(TRAIN_PATH,'../',PAIRS_FILE),
                       help = 'Pairs csv file path')

    args = parser.parse_args()
    
    train(n_epochs = args.n_epochs, pairs_file = args.pairs_file, test_users = args.test_users)