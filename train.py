import numpy as np
import pandas as pd
import h5py
import glob

from sklearn.model_selection import train_test_split

import torch
from torch.autograd import Variable
import torch.utils.data as data

from unet import UNet
from low_pass_filter import low_pass_filter

class SleepApneaDataset(torch.utils.data.Dataset):

    def __init__(self, X_PATH, Y_PATH):
        self.file = h5py.File(X_PATH,'r')
        self.labels = pd.read_csv(Y_PATH)

    def __len__(self):
        return self.file['data'].shape[0]

    def __getitem__(self, index):
        X = np.array(self.file.get('data'))
        y = self.labels[index]
        return (X[index, 2:], y)


if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # Loading data
    X_TRAIN_LABELS_PATH = "./data/X_train_h7ipJUo.csv"
    Y_TRAIN_PATH = "./data/y_train_tX9Br0C.csv"
    X_TRAIN_PATH = "./data/X_train.h5"

    X_train_labels = pd.read_csv(X_TRAIN_LABELS_PATH)
    train_dataset = h5py.File(X_TRAIN_PATH)
    y_train = pd.read_csv(Y_TRAIN_PATH)

    # First to columns are removed
    X_train = train_dataset['data'][:, 2:]

    # Reshape to a dataset with 8 channels in 3rd dimension
    X_train = X_train.reshape(4400, 8, 9000)

    # Application of a low pass filter to 10Hz to remove noise
    # Cut off frequency to play with
    # X_train = low_pass_filter(X_train, 10) # Not working

    # Train test split (could be improved in k fold validation)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size=0.8, random_state=42)

    train_set = SleepApneaDataset(X_TRAIN_PATH, Y_TRAIN_PATH)

    BATCH_SIZE = 220

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4
    )

    val_loader = torch.utils.data.DataLoader(
        dataset=X_val,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4
    )

    # Def of network
    model = UNet(in_channels=8, out_channels=1)
    model.to(device)

    # Optimizer : Adam
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # Loss function Cross Entropy (behind sigmoid layers)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Training :
    for epoch in range(10):
        model.train()
        for batch_idx, (x, target) in enumerate(train_loader):
            optimizer.zero_grad()
            x, target = Variable(x).to(device), Variable(target).to(device)
            out = model(x)
            loss = loss_fn(out, target)
            loss.backward()
            optimizer.step()

            # Print performances every 50 batches
            if batch_idx % 50 == 0:
                print(f" Batch {batch_idx}, Loss : {loss}")

        # testing
        model.eval()
        correct = 0

        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(val_loader):
                x, target = x.to(device), target.to(device)
                out = model(x)
                loss = loss_fn(out, target)
                # _, prediction = torch.max(out.data, 1)
                prediction = out.argmax(dim=1, keepdim=True) # index of the max log-probability
                correct += prediction.eq(target.view_as(prediction)).sum().item()
        taux_classif = 100. * correct / len(test_loader.dataset)
        print('Accuracy: {}/{} (tx {:.2f}%, err {:.2f}%)\n'.format(correct,
        len(val_loader.dataset), taux_classif, 100.-taux_classif))

