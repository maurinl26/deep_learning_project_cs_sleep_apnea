from scipy.signal import butter, sosfilt
import numpy as np
import pandas as pd

from visualisation import visualize_signal_and_event


def low_pass_filter(sig, freq=10):
    """
    Apply a low pass filter to data to remove noise

    :param sig: given signals, a numpy ndarray
    :param freq: cut off frequency of low-pass filter
    :return: a ndarray containing filtered signals
    """
    sos = butter(2, 10, fs=100, btype='lp', output='sos')
    filtered = sosfilt(sos, sig, axis=1)
    return filtered


if __name__ == "__main__":
    SIGNALS_NAME = [
        "AbdoBelt",
        "AirFlow",
        "PPG",
        "ThorBelt",
        "Snoring",
        "SPO2",
        "C4A1",
        "O2A1"
    ]

    # Load data
    X_TRAIN_PATH = "./data/X_train.npy"
    train_file = np.load(X_TRAIN_PATH)
    X_train = train_file[:, 2:]

    # Load masks
    PATH_TO_TRAINING_TARGET = './data/y_train_tX9Br0C.csv'
    mask = np.array(pd.read_csv(PATH_TO_TRAINING_TARGET))

    print(X_train.shape)

    X = X_train.reshape(4400, 8, 9000)

    # Plot of raw signals
    x = X[1, :, :]
    visualize_signal_and_event(x, mask[1, :], signals_name=SIGNALS_NAME, signal_freq=100)

    # Test of low pass filtering
    X_filtered = low_pass_filter(X_train, 10)
    X_filtered = X_filtered.reshape(4400, 8, 9000)

    # Plot of filtered signals
    x = X_filtered[1, :, :]
    visualize_signal_and_event(x, mask[1, :], signals_name=SIGNALS_NAME, signal_freq=100)








