import h5py
import pandas as pd
import datetime as dt

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from scipy import signal

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, f1_score
from sklearn.preprocessing import Normalizer

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import metric_dreem

# Creating data loaders
class Dataset(torch.utils.data.Dataset):

    def __init__(self, sample, sample_fft, labels):
        self.sample = sample
        self.sample_fft = sample_fft
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.sample[index], self.sample_fft[index], self.labels[index]


# use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Amount of data to load (max: 4400)
n_data = 4400

# Loading data
print('Loading data ...')
PATH_TO_TRAINING_DATA = "./data/X_train.h5"
PATH_TO_TRAINING_TARGET = "./data/y_train_tX9Br0C.csv"
h5_file = h5py.File(PATH_TO_TRAINING_DATA, 'r')

# mask represents y, the results
mask = np.array(pd.read_csv(PATH_TO_TRAINING_TARGET))
file = h5py.File(PATH_TO_TRAINING_DATA, 'r')
data = file['data']

# Separating the different signals
print('Reshaping data ...')
N_signals = 8
x = data[:n_data, 2:]
x = x.reshape(n_data, N_signals, -1)
# x = np.transpose(x, (0, 2, 1))  # shape = (n_data, 9000, 8)
mask = np.array(pd.read_csv(PATH_TO_TRAINING_TARGET))[
    :n_data, 1:]  # shape = (n_data, 90)

# Validation split
X_train, X_val, y_train, y_val = train_test_split(
    x, mask, test_size=0.2, random_state=1)

def normalize(x_train, x_val):
    mean_ = x_train.mean(axis=0).mean(axis=0)
    std_ = x_train.std(axis=0).mean(axis=0)
    x_train = (x_train-mean_)/std_
    x_val = (x_val-mean_)/std_
    return x_train, x_val

X_train, X_val = normalize(X_train, X_val)

def get_fft(X, channel):  # Get the Fourier transform for one of the 8 signals
    fs = 100
    d = X[:, channel, :]
    Sxx = signal.spectrogram(d, fs, nperseg=1024, noverlap=935)[2]
    Sxx = np.clip(Sxx, -10, 10)[:, :15, :]/5-1
    Sxx = np.transpose(Sxx, (0, 2, 1))
    return Sxx

# The channels where extracting the Fourier transform is relevant
channels = [0, 1, 2, 3, 6, 7]

def get_fft_inputs(X):  # Concatenate the FFT signals
    ffts = [get_fft(X, c) for c in channels]
    X_new = np.concatenate(ffts, axis=-1)
    return X_new

# Create FFT signals from the original signals
print('Calculating fft ...')
X_train_fft = get_fft_inputs(X_train)
X_val_fft = get_fft_inputs(X_val)

BATCH_SIZE = 64

train_loader = torch.utils.data.DataLoader(
    dataset=Dataset(X_train, X_train_fft, y_train),
    batch_size=BATCH_SIZE,
    shuffle=True,
)

val_loader = torch.utils.data.DataLoader(
    dataset=Dataset(X_val, X_val_fft, y_val),
    batch_size=BATCH_SIZE,
    shuffle=True,
)

EPSILON = 1e-10
def recall(y_true, y_pred):
    true_positives = torch.sum(torch.round(torch.clip(y_true * y_pred, 0, 1)))
    possible_positives = torch.sum(torch.round(torch.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + EPSILON)
    return recall

def precision(y_true, y_pred):
    true_positives = torch.sum(torch.round(torch.clip(y_true * y_pred, 0, 1)))
    predicted_positives = torch.sum(torch.round(torch.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + EPSILON)
    return precision

def f1(y_true, y_pred):  # A function to calculate the F1 score       
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+EPSILON))


N_FILTERS1 = 20
KERNEL_SIZE = 21

N_FILTERS2 = 40

N_FILTERS3 = 20

N_FILTERS4 = 40

KERNEL_SIZE5 = 11

# CNN 1D network for X_true
print("Creating model ...")
class CNNet(nn.Module):

    def __init__(self):
        super(CNNet, self).__init__()
        # 1D Convolutional Net for X -> input size (9000, 8)
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=8, out_channels=N_FILTERS1, kernel_size=KERNEL_SIZE, stride=1, padding=10), # use same padding,
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool1d(10)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(N_FILTERS1, N_FILTERS2, KERNEL_SIZE, stride=1, padding=10), # kernel_size = 5,
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool1d(10)
        )

        # 1D Convolutional Net for X_fft -> input size (90, 90)
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=90, out_channels=N_FILTERS3, kernel_size=KERNEL_SIZE, stride=1, padding=10),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.conv4 = nn.Conv1d(in_channels=N_FILTERS2+N_FILTERS3, out_channels=N_FILTERS4, kernel_size=KERNEL_SIZE, stride=1, padding=10)
        self.conv5 = nn.Conv1d(in_channels=N_FILTERS4, out_channels=1, kernel_size=KERNEL_SIZE5, stride=1, padding=5)


    def forward(self, x, x_fft):
        x = self.conv1(x)
        x = self.conv2(x)

        x_fft = self.conv3(x_fft)

        x_combined = torch.cat((x, x_fft), dim=1)
        x_combined = F.relu(self.conv4(x_combined))
        x_combined = self.conv5(x_combined) # no activation function for use with BCEWithLogitsLoss
        return x_combined.squeeze()

print('Starting training')
model = CNNet()
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
loss_fn = nn.BCEWithLogitsLoss()

NUM_EPOCHS = 100
for epoch in range(NUM_EPOCHS):
    total_loss = 0
    # training
    model.train() # mode "train" agit sur "dropout" ou "batchnorm"
    for batch_idx, (x, x_fft, target) in enumerate(train_loader):
        optimizer.zero_grad()
        x, x_fft, target = Variable(x).to(device), Variable(x_fft).to(device),Variable(target).to(device)
        out = model(x.float(), x_fft.float())
        loss = loss_fn(out, target.float())
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print('epoch [{}/{}] training loss: {}'.format(epoch+1, NUM_EPOCHS, total_loss/len(train_loader)))

    if (epoch + 1) % 10 == 0:
        # testing every 10 epochs
        model.eval()
        val_loss = 0
        score = 0
        with torch.no_grad():
            for batch_idx, (x, x_fft, target) in enumerate(val_loader):
                x, x_fft, target = Variable(x).to(device), Variable(x_fft).to(device),Variable(target).to(device)
                out = model(x.float(), x_fft.float())
                loss = loss_fn(out, target.float())

                out, target = out.cpu(), target.cpu()
                score += metric_dreem.dreem_sleep_apnea_custom_metric(out, target)
                val_loss += loss.item()

        final_score = 100. * score / len(val_loader.dataset)
        print('Validation loss: {}/{})\n'.format(val_loss/len(val_loader.dataset), len(val_loader.dataset)))


now_str = dt.datetime.now().strftime('%m-%d-%Y_%H-%M')
torch.save(model, './model/'+now_str+'_cnn')
