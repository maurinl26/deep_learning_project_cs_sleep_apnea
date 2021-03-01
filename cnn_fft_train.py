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
from torch.autograd import Variable
from cnn_fft_model import CNNet

import metric_dreem
from loss import IoULoss, DiceLoss

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
    true_positives = torch.sum(torch.round(torch.clip(y_true * y_pred, 0, 1).float()))
    possible_positives = torch.sum(torch.round(torch.clip(y_true, 0, 1).float()))
    rec = true_positives / (possible_positives + EPSILON)
    return rec

def precision(y_true, y_pred):
    true_positives = torch.sum(torch.round(torch.clip(y_true * y_pred, 0, 1).float()))
    predicted_positives = torch.sum(torch.round(torch.clip(y_pred, 0, 1).float()))
    prec = true_positives / (predicted_positives + EPSILON)
    return prec

def f1(y_true, y_pred):  # A function to calculate the F1 score
    y_true, y_pred = torch.sigmoid(y_true), torch.sigmoid(y_pred) # we add the sigmoid here because we use BCEWithLogitsLoss
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return 2*((prec*rec)/(prec+rec+EPSILON))

print("Creating model ...")
load_model=False # if True, loads an existing model and continues training it
MODEL_PATH = './model/02-27-2021_23-25_cnn'
model = CNNet()
if load_model:
    model = torch.load(MODEL_PATH)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
loss_fn = nn.BCEWithLogitsLoss()


# Training
print('Starting training')

min_val_loss = np.inf
max_val_f1 = 0
now_str = dt.datetime.now().strftime('%m-%d-%Y_%H-%M')

NUM_EPOCHS = 300
for epoch in range(NUM_EPOCHS):
    total_loss = 0
    total_f1 = 0
    model.train()

    for batch_idx, (x, x_fft, target) in enumerate(train_loader):
        optimizer.zero_grad()
        x, x_fft, target = Variable(x).to(device), Variable(x_fft).to(device), Variable(target).to(device)
        out = model(x.float(), x_fft.float())
        loss = loss_fn(out, target.float())
        loss.backward()
        optimizer.step()

        # update metrics
        f1_score = f1(out, target.float())
        total_f1 +=  f1_score
        total_loss += loss.item()
    
    # display metrics
    total_loss = total_loss/len(train_loader)
    total_f1 = total_f1/len(train_loader)
    print(f'epoch [{epoch+1}/{NUM_EPOCHS}] training loss: {total_loss:.4} F1 score: {total_f1:.4}')

    if (epoch + 1) % 10 == 0: # testing with validation set every 10 epochs
        model.eval()
        val_loss = 0
        val_f1 = 0
        score = 0
        with torch.no_grad():
            for batch_idx, (x, x_fft, target) in enumerate(val_loader):
                x, x_fft, target = Variable(x).to(device), Variable(x_fft).to(device),Variable(target).to(device)
                out = model(x.float(), x_fft.float())
                loss = loss_fn(out, target.float())

                # update metrics
                out, target = out.cpu(), target.cpu()
                val_loss += loss.item()
                val_f1 += f1(out, target.float())

        # display metrics
        val_loss = val_loss/len(val_loader)
        val_f1 = val_f1/len(val_loader)
        print(f'Validation loss: {val_loss:.4} Validation F1 score: {val_f1:.4}')

        # save model if validation loss has decreased
        if val_f1 > max_val_f1:
            print('Saving model ...')
            max_val_f1 = val_f1
            torch.save(model, './model/'+now_str+'_cnn')
        print()
