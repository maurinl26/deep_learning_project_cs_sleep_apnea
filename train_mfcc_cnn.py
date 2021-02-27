import time

import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchaudio import transforms

from loss import IoULoss


class Dataset(torch.utils.data.Dataset):
    def __init__(self, samples, labels):
        self.samples = samples
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.samples[index], self.labels[index]


class CNN(nn.Module):
    def __init__(self, n_channels):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(n_channels, 10, kernel_size=10, padding=(5, 5)),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(10, 20, kernel_size=10, padding=(4, 4)),
            nn.ReLU(inplace=True)
        )

        self.segment = nn.Sequential(nn.Conv1d(20, 1, kernel_size=1), nn.Sigmoid())

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.mean(x, dim=3)
        x = self.segment(x)

        return x


if __name__ == "__main__":

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # Loading data
    print("Loading data ...")
    X_TRAIN_LABELS_PATH = "./data/X_train_h7ipJUo.csv"
    Y_TRAIN_PATH = "./data/y_train_tX9Br0C.csv"
    X_TRAIN_PATH = "./data/X_train.h5"

    X_labels = pd.read_csv(X_TRAIN_LABELS_PATH).to_numpy()
    train_file = h5py.File(X_TRAIN_PATH, "r")
    X = torch.from_numpy(np.array(train_file["data"])).float()
    y = torch.from_numpy(pd.read_csv(Y_TRAIN_PATH).to_numpy()).float()

    # First to columns are removed
    X = X[:, 2:]
    y_train = y[:, 1:]

    # Reshape to a dataset with 8 channels in 2nd dimension
    X = X.view(4400, 8, 9000)

    # Keep only snoring
    X_train = X[:, :, :]

    # Apply MFCC features
    print("Features extraction with MFCC... ")
    X_train = transforms.MFCC(sample_rate=100, n_mfcc=90)(X_train)
    x, y, z = X_train.shape

    #X_train = X_train.view(x, 1, y, z)
    print(f"X_train shape : {X_train.shape}")

    # Train test split (could be improved in k fold validation)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, train_size=0.8, random_state=42
    )

    train_set = Dataset(X_train, y_train)
    val_set = Dataset(X_val, y_val)

    # Hyperparameters
    BATCH_SIZE = 64
    NUM_EPOCHS = 50
    print(f"Hyperparameters : Batch_size = {BATCH_SIZE}, Num of epochs = {NUM_EPOCHS}")

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=BATCH_SIZE, shuffle=True
    )

    val_loader = torch.utils.data.DataLoader(
        dataset=val_set, batch_size=BATCH_SIZE, shuffle=False
    )

    model = CNN(n_channels=8).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = IoULoss()

    # Train model
    total_step = len(train_loader)
    start = time.time()
    epoch_losses = []
    print("Start training ...")
    for epoch in range(NUM_EPOCHS):
        losses = []
        model.train()
        for i, (img, lab) in enumerate(train_loader):
            outputs = model(img)
            loss = loss_fn(outputs, lab)
            losses.append(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % BATCH_SIZE == 1:
                print(
                    f"Epoch [{epoch}/{NUM_EPOCHS}], Loss: {loss}"
                )

        epoch_losses.append(np.sum(losses) / len(losses))
