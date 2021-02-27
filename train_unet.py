import numpy as np
import pandas as pd
import datetime as dt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import torch
from torch.autograd import Variable
import torch.utils.data as data

from unet import UNet
from loss import IoULoss, DiceLoss
from low_pass_filter import low_pass_filter

# DataLoader
class Dataset(torch.utils.data.Dataset):

    def __init__(self, labels, samples):
        self.labels = labels
        self.samples = samples

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.samples[index], self.labels[index]


if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # Loading data
    X_TRAIN_LABELS_PATH = "./data/X_train_h7ipJUo.csv"
    Y_TRAIN_PATH = "./data/y_train_tX9Br0C.csv"
    X_TRAIN_PATH = "./data/X_train_scaled.npy"

    X_labels = pd.read_csv(X_TRAIN_LABELS_PATH).to_numpy()
    X = torch.from_numpy(np.load(X_TRAIN_PATH)).float()
    y = torch.from_numpy(pd.read_csv(Y_TRAIN_PATH).to_numpy()).long()

    # First to columns are removed
    X_train = X[:, 2:]
    y_train = y[:, 1:]

    # Reshape to a dataset with 8 channels in 3rd dimension
    X_train = X_train.reshape(4400, 8, 9000)

    # Select Signals
    X_train = X_train[:, 1:4, :]
    print(f" Taille du dataset {X_train.shape}")

    # Train test split (could be improved in k fold validation)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size=0.8, random_state=42)

    # train_set = SleepApneaDataset(X_TRAIN_PATH, Y_TRAIN_PATH)
    BATCH_SIZE = 128

    train_loader = torch.utils.data.DataLoader(
        dataset=Dataset(y_train, X_train),
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

    CHANNELS = 3
    N_FEATURES = 9000

    # Def of network
    model = UNet(CHANNELS, N_FEATURES)
    model.to(device)

    # Optimizer : Adam
    #optimizer = torch.optim.SGD(model.parameters(), lr=1e-6, momentum=0.9, weight_decay=5e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-6)

    # Loss function Cross Entropy (behind sigmoid layers)
    loss_fn = torch.nn.CrossEntropyLoss()
    loss_fn = IoULoss()

    N_EPOCHS = 10

    # Training :
    for epoch in range(N_EPOCHS):
        model.train()
        for batch_idx, (x, target) in enumerate(train_loader):
            optimizer.zero_grad()
            x, target = Variable(x).to(device), Variable(target).to(device)
            out = model(x)
            loss = loss_fn.forward(out, target)
            loss.backward()
            optimizer.step()

            # Print performances every 50 batches
            if batch_idx % 64 == 0:
                print(f" Epoch : {epoch}, Batch {batch_idx}, Loss : {loss}")


        """
        # testing
        model.eval()
        correct = 0
        """

        """
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(val_loader):
                x, target = x.to(device), target.to(device)
                out = model(x)
                loss = loss_fn(out, target)
                # _, prediction = torch.max(out.data, 1)
                prediction = out.argmax(dim=1, keepdim=True) # index of the max log-probability
                correct += prediction.eq(target.view_as(prediction)).sum().item()
        taux_classif = 100. * correct / len(val_loader.dataset)
        print('Accuracy: {}/{} (tx {:.2f}%, err {:.2f}%)\n'.format(correct,
        len(val_loader.dataset), taux_classif, 100.-taux_classif))"
        """

