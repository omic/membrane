#!/usr/bin/env python
# coding: utf-8
from utils import *
from network import *

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
            loss.backward()
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




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-users', nargs='*', default = None,
                       help="Leave out test sets from train sets")#, default = 'data/wav/enroll/19-enroll.wav')
    parser.add_argument('--n-epochs', default = 1,
                       help = 'Set the number of epochs')
    parser.add_argument('--pairs-file', default = os.path.join(TRAIN_PATH,'../',PAIRS_FILE),
                       help = 'Pairs csv file path')
    args = parser.parse_args()
    train(n_epochs = args.n_epochs, pairs_file = args.pairs_file, test_users = args.test_users)
