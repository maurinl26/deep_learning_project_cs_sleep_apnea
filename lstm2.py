import numpy as np
import pandas as pd
import h5py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import time
from sklearn.model_selection import train_test_split

from loss import IoULoss

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device = {}'.format(device))

# Hyper-parameters
sequence_length = 9000 # we divide the sequence into 90 periods
input_size = 8
hidden_size = 5
num_layers = 1
num_classes = 10
batch_size = 5
num_epochs = 30
learning_rate = 0.01

# DataLoader
class Dataset(torch.utils.data.Dataset):

    def __init__(self, samples, labels):
        self.samples = samples
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.samples[index], self.labels[index]

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

# Reshape to an 8-dimensional vector sampled 9000 times
X_train = X_train.reshape(4400, 8, 9000)
X_train = X_train.permute(0, 2, 1) #output size is (4400, 9000, 8)

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
        sequence2_size = 90
        in_size2 = 100
        hidden_size2 = 10
        nb_layer2 = 1
        
        super(LSTMNet,self).__init__()
        self.hidden_size = hidden_size
        self.hidden_size2 = hidden_size2
        self.nb_layer = nb_layer
        self.nb_layer2 = nb_layer2
        self.lstm = nn.LSTM(in_size, hidden_size, nb_layer, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1) # single output for BCELoss
        self.lstm2 = nn.LSTM(in_size2, hidden_size2, nb_layer2, batch_first=True)
        self.fc2 = nn.Linear(hidden_size2, 1)

    def forward(self, x):
        # initial states
        h0 = torch.zeros(self.nb_layer, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.nb_layer, x.size(0), self.hidden_size).to(device)

        h1 = torch.zeros(self.nb_layer2, x.size(0), self.hidden_size2).to(device)
        c1 = torch.zeros(self.nb_layer2, x.size(0), self.hidden_size2).to(device)

        out, _ = self.lstm(x, (h0,c0))
        out = self.fc(out)
        # print('1:', out.size())
        out = out.reshape(-1, 90, 100)
        # print('2:', out.size())
        out, _ = self.lstm2(out, (h1,c1))
        out = self.fc2(out)
        # print('3:', out.size())
        return out.squeeze(2) # change shape from (1, 90, 1) to (1, 90)


model = LSTMNet(input_size, hidden_size, num_layers).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)
loss_fn = IoULoss()

checkpoint = torch.load('./model/lstm2.pt')
model.load_state_dict(checkpoint['model_state_dict'])
# # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# # epoch = checkpoint['epoch']
# # loss = checkpoint['loss']


# training
total_step = len(train_loader)
start = time.time()
epoch_losses = []
val_losses = []
min_val_loss = np.inf

print('Starting training ...')
for epoch in range(num_epochs):
    model.train()
    losses = []
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

    # evaluate validation loss
    model.eval()
    val_loss = []
    for i, (img, lab) in enumerate(val_loader):
        img = img.reshape(-1, sequence_length, input_size).to(device)
        lab = lab.to(device)

        outputs = model(img)
        loss = loss_fn(outputs, lab)
        val_loss.append(loss)
    val_losses.append(np.sum(val_loss)/len(val_loss))

    # save model if it does better on the validation set
    if val_losses[-1] < min_val_loss:
        # now_str = dt.datetime.now().strftime('%m-%d%-Y')
        torch.save({'model_state_dict': model.state_dict()}, './model/lstm2.pt')


import matplotlib.pyplot as plt

x = np.arange(len(epoch_losses))
plt.plot(x, epoch_losses)
plt.show()
