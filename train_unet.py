import datetime as dt
import logging

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as data
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.autograd import Variable

from loss import DiceLoss, IoULoss
from metric_dreem import dreem_sleep_apnea_custom_metric
from unet import UNet


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
    now_str = dt.datetime.now().strftime("%m-%d-%Y_%H-%M")
    logging.basicConfig(
        filename=f"./logs/{now_str}_UNet_training.log",
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p",
    )

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device : {device}")

    # Loading data
    logging.info("Loading data ...")
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
    X_train = X.view(4400, 8, 9000)

    # Train test split (could be improved in k fold validation)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, train_size=0.8, random_state=42
    )

    # train_set = SleepApneaDataset(X_TRAIN_PATH, Y_TRAIN_PATH)
    BATCH_SIZE = 128
    CHANNELS = 8
    DROPOUT = 0.1
    N_FEATURES = 9000
    N_EPOCHS = 75
    LEARNING_RATE = 0.001

    logging.info(
        f"Hyperparameters : "
        f"Batch size : {BATCH_SIZE}, "
        f"Epochs : {N_EPOCHS}, "
        f"Learning rate : {LEARNING_RATE}, "
        f"Dropout : {DROPOUT}"
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=Dataset(y_train, X_train),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
    )

    val_loader = torch.utils.data.DataLoader(
        dataset=Dataset(y_val, X_val),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
    )

    # Def of network
    model = UNet(CHANNELS, DROPOUT)
    model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Number of parameters for UNet : {n_params}")

    # Optimizer : Adam
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Loss function
    loss_fn = nn.BCEWithLogitsLoss()
    # loss_fn = DiceLoss()
    # loss_fn = IoULoss()

    # Training :
    for epoch in range(N_EPOCHS):
        model.train()
        for batch_idx, (x, target) in enumerate(train_loader):
            optimizer.zero_grad()
            x, target = Variable(x).to(device), Variable(target).to(device)
            out = model(x)
            out = torch.squeeze(out)
            loss = loss_fn.forward(out, target)
            loss.backward()
            optimizer.step()

            # Print performances every 50 batches
            if batch_idx % 64 == 0:
                print(f"Epoch : {epoch}, Batch {batch_idx}, Loss : {loss}")
                logging.info(f"Epoch : {epoch}, Batch {batch_idx}, Loss : {loss}")

        # testing
        model.eval()
        loss_val = 0
        min_loss = 2

        THRESHOLDS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        dreem_metrics = [0 for i in range(len(THRESHOLDS))]

        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(val_loader):
                x, target = x.to(device), target.to(device)
                out = model(x)
                out = torch.squeeze(out)
                loss = loss_fn(out, target)

                loss_val += loss.item()

                for i, t in enumerate(THRESHOLDS):
                    out = out > t
                    out = out.int()

                    out, target = out.cpu(), target.cpu()
                    dreem_metrics[i] += dreem_sleep_apnea_custom_metric(out, target)

            print(f"Validation loss: {loss_val / len(val_loader)}")
            logging.info(f"Validation loss: {loss_val / len(val_loader)}")

            for i, val in enumerate(dreem_metrics):
                print(
                    f"Threshold : {THRESHOLDS[i]}, Dreem metric : {val / len(val_loader)}"
                )
                logging.info(
                    f"Threshold : {THRESHOLDS[i]}, Dreem metric : {val / len(val_loader)}"
                )

        if min_loss > loss_val:
            best_params = model.state_dict()
            min_loss = loss_val

    now_str = dt.datetime.now().strftime("%m-%d-%Y_%H-%M")
    torch.save(model.state_dict, "./model/" + now_str + "_unet")
