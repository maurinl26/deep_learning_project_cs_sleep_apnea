from scipy.signal import butter, sosfilt
import numpy as np
import pandas as pd
import h5py

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
    X_TRAIN_PATH = "./data/X_train.h5"

    data = h5py.File(X_TRAIN_PATH)
    time_series = np.array(data.get('data'))

    time_series = MinMaxScaler().fit_transform(time_series)
    print(time_series.shape)

    #np.save('X_train_scaled.npy', time_series)








