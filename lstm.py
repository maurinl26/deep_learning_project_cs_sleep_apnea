import numpy as np
import pandas as pd
import h5py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import time
from sklearn.model_selection import train_test_split

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device = {}'.format(device))

# Hyper-parameters
sequence_length = 90 # we divide the sequence into 90 periods
input_size = 8*100
hidden_size = 1000
num_layers = 1
num_classes = 10
batch_size = 10
num_epochs = 20
learning_rate = 0.001

# DataLoader
class Dataset(torch.utils.data.Dataset):

    def __init__(self, samples, labels):
        self.samples = samples
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.samples[index], self.labels[index]

def reshape_data(X):
    # transforms the 90 second samples into frames of 1 second
    out = torch.zeros(X.size(0), 90, 800)
    for i in range(90):
        a = X[:, :, i, :].reshape(X.size(0), 800)
        out[:, i, :] = a
    return out

# Loading data
print('Loading data ...')
X_TRAIN_LABELS_PATH = "./data/X_train_h7ipJUo.csv"
Y_TRAIN_PATH = "./data/y_train_tX9Br0C.csv"
X_TRAIN_PATH = "./data/X_train.h5"

X_labels = pd.read_csv(X_TRAIN_LABELS_PATH).to_numpy()
train_file = h5py.File(X_TRAIN_PATH, 'r')
X = torch.from_numpy(np.array(train_file['data'])).float()
y = torch.from_numpy(pd.read_csv(Y_TRAIN_PATH).to_numpy()).float()

# First to columns are removed
X_train = X[:, 2:]
y_train = y[:, 1:]

# Reshape to a dataset with 8 channels in 2nd dimension
X_train = X_train.reshape(4400, 8, 90, 100)
X_train = reshape_data(X_train) # shape is (4400, 90, 800)

# Train test split (could be improved in k fold validation)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size=0.8, random_state=42)

train_set = Dataset(X_train, y_train)
val_set = Dataset(X_val, y_val)

train_loader = torch.utils.data.DataLoader(
                 dataset=train_set,
                 batch_size=batch_size,
                 shuffle=True)

val_loader = torch.utils.data.DataLoader(
                dataset=val_set,
                batch_size=batch_size,
                shuffle=False)

# define LSTM model
class LSTMNet(nn.Module):
    def __init__(self, in_size, hidden_size, nb_layer):
        super(LSTMNet,self).__init__()
        self.hidden_size = hidden_size
        self.nb_layer = nb_layer
        self.lstm = nn.LSTM(in_size, hidden_size, nb_layer, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1) # single output for BCELoss

    def forward(self,x):
        # initial states
        h0 = torch.zeros(self.nb_layer, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.nb_layer, x.size(0), self.hidden_size).to(device)

        out,_ = self.lstm(x, (h0,c0))
        out = self.fc(out)
        return out.squeeze(2) # reshape from size (1, 90, 1) to (1, 90)


model = LSTMNet(input_size, hidden_size, num_layers).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
loss_fn = nn.BCEWithLogitsLoss()

# training
total_step = len(train_loader)
start = time.time()
epoch_losses = []
print('Starting training ...')
for epoch in range(num_epochs):
    losses = []
    model.train()
    for i, (img, lab) in enumerate(train_loader):
        img = img.reshape(-1, sequence_length, input_size).to(device)
        lab = lab.to(device)

        outputs = model(img)
        loss = loss_fn(outputs, lab)
        losses.append(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f} ({:.2f} s)'
            .format(epoch+1, num_epochs, i+1, total_step,
            loss.item(), time.time()-start))
    epoch_losses.append(np.sum(losses)/len(losses))

# test
# with torch.no_grad():
#     correct = 0
#     total = 0
#     for img, lab in val_loader:
#         img = img.reshape(-1, sequence_length, input_size).to(device)
#         lab = lab.to(device)
#         outputs = model(img)
#         total += lab.size(0)
#         correct += (pred == lab).sum().item()

#     print('Test Accuracy: {}%'.format(100. * correct / total) )


import matplotlib.pyplot as plt

y = np.arange(len(epoch_losses))
plt.plot(epoch_losses, y)
plt.show()
