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
# X_train_fft = get_fft_inputs(X_train)
X_val_fft = get_fft_inputs(X_val)

BATCH_SIZE = 40

# train_loader = torch.utils.data.DataLoader(
#     dataset=Dataset(X_train, X_train_fft, y_train),
#     batch_size=BATCH_SIZE,
#     shuffle=True,
# )

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
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return prec, rec, 2*((prec*rec)/(prec+rec+EPSILON))

print("Creating model ...")
load_model=True
MODEL_PATH = './model/02-28-2021_22-51_cnn'
model = CNNet()
if load_model:
    model = torch.load(MODEL_PATH)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
loss_fn = nn.BCEWithLogitsLoss()


# Training
print('Evaluation :')
model.eval()

thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
with torch.no_grad():

    for t in thresholds:
        print(f'Threshold = {t}')
        val_loss = 0
        val_f1 = 0
        val_acc = 0
        val_rec = 0
        score = 0

        for batch_idx, (x, x_fft, target) in enumerate(val_loader):
            x, x_fft, target = Variable(x).to(device), Variable(x_fft).to(device), Variable(target).to(device)
            # out = torch.rand(target.size()).to(device) # comparison to random predictions
            out = model(x.float(), x_fft.float())
            
            out = torch.sigmoid(out)
            out = out > t
            out = out.int()

            out, target = out.cpu(), target.cpu()
            score += metric_dreem.dreem_sleep_apnea_custom_metric(out, target)
            acc, rec, f1_score = f1(out, target)
            val_acc += acc
            val_rec += rec
            val_f1 += f1_score

        n = len(val_loader)
        final_score = 100. * score / len(val_loader.dataset)
        print(f'F1 score: {val_f1/n:.4}')
        print(f'Accuracy: {val_acc/n:.4}')
        print(f'Recall: {val_rec/n:.4}')
        print(f'Dreem score: {final_score:.4}')
        print()

