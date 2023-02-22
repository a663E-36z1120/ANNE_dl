# Code for integrating raw data into appropriate formats for later use
# @author: Andrew H. Zhang

import numpy as np
from pyedflib import highlevel
from matplotlib import pyplot as plt

DATA_DIR = "/mnt/Common/Data"
SAMP_RATE = 25  # Hz
WINDOW_LEN = 30  # Seconds


def integrate_data(subject_id):
    """ Prepares the data tensor and target vector of subject with subject_id
    :param subject_id: the id of the subject for whom the data tensor and target will be prepared
    :return: X: 3d tensor in the shape of (n, s, c), where:
    n is the number of 30-second windows in the time-series data set
    s the number of signal samples in each 30-second window (s = sample_rate * 30 seconds)
    c is the number of signal sources, preliminarirly:
    0: ECG  1: PPG  2,3,4: x,y,z accelerometer  5: ENMO 6: z-angle  7: chest temperature
    """
    # Read raw signals from csv
    index, time, ecg, ppg, x_acc, y_acc, z_acc, enmo, z_angle, temp \
        = np.loadtxt(f"{DATA_DIR}/{subject_id}/ANNE.csv", delimiter=",", dtype="float",
                     skiprows=1, unpack=True,
                     usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9))
    n = len(ecg) // (SAMP_RATE*WINDOW_LEN) * (SAMP_RATE*WINDOW_LEN)

    # Features
    ecg = np.reshape(ecg[:n], (-1, SAMP_RATE*WINDOW_LEN, 1))
    ppg = np.reshape(ppg[:n], (-1, SAMP_RATE*WINDOW_LEN, 1))
    x_acc = np.reshape(x_acc[:n], (-1, SAMP_RATE*WINDOW_LEN, 1))
    y_acc = np.reshape(y_acc[:n], (-1, SAMP_RATE*WINDOW_LEN, 1))
    z_acc = np.reshape(z_acc[:n], (-1, SAMP_RATE*WINDOW_LEN, 1))
    enmo = np.reshape(enmo[:n], (-1, SAMP_RATE*WINDOW_LEN, 1))
    z_angle = np.reshape(z_angle[:n], (-1, SAMP_RATE*WINDOW_LEN, 1))
    temp = np.reshape(temp[:n], (-1, SAMP_RATE*WINDOW_LEN, 1))

    X = np.concatenate((ecg, ppg, x_acc, y_acc, z_acc, enmo, z_angle, temp), axis=2)

    # targets
    t = np.loadtxt(f"{DATA_DIR}/{subject_id}/PSG_score.txt", dtype="str")
    t = np.reshape(t, (-1, len(t)))

    return X, t


def plot_window(X, index=0):
    """
    Plot the index-th 30 second window in the data tensor X
    """
    fig, axs = plt.subplots(X.shape[-1])
    for i in range(len(axs)):
        arr = X[index, :, i]
        axs[i].plot(arr)
    plt.show()


if __name__ == "__main__":
    X, t = integrate_data(132)
    plot_window(X, index=80)