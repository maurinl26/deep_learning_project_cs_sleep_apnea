import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.model_selection import train_test_split

from dataset import Dataset
from unet import UNet
from metric_dreem import dreem_sleep_apnea_custom_metric


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet(channels=8)
    MODEL_PATH = "./model/03-01-2021_13-57_unet"
    model = torch.load(MODEL_PATH)
    model.to(device)

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
    X_train = X.view(4400, 8, 9000)

    # Train test split (could be improved in k fold validation)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, train_size=0.8, random_state=42
    )

    BATCH_SIZE = 128

    val_loader = torch.utils.data.DataLoader(
        dataset=Dataset(y_val, X_val),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
    )

    model.eval()
    loss_val = 0

    threshold_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    dreem_metric_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(val_loader):
            x, target = x.to(device), target.to(device)
            out = model(x)
            out = torch.squeeze(out)
            loss = loss_fn(out, target)
            loss_val += loss.item()

            for i, THRESHOLD in enumerate(threshold_list):
                out = out > THRESHOLD
                out = out.int()

                dreem_metric_list[i] += dreem_sleep_apnea_custom_metric(out, target)

    for i, metric in enumerate(dreem_metric_list):
        print(
            f" Threshold value : {threshold_list[i]}, Dreem metric : {dreem_metric_list[i] / len(val_loader)}"
        )
