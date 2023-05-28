# Code for integrating raw data into appropriate formats for later use
# @author: Andrew H. Zhang

import numpy as np
from pyedflib import highlevel
from matplotlib import pyplot as plt
from math import ceil
from scipy import signal
from feature_engineering.features import *

DATA_DIR = "/mnt/Common/Data"
SAMP_RATE = 25  # Hz
WINDOW_LEN = 30  # Seconds


def integrate_data(subject_id, shift=0):
    """ Prepares the data tensor and target vector of subject with subject_id
    :param subject_id: the id of the subject for whom the data tensor and target will be prepared
    :param shift: shift of the ANNE signal in seconds, -: to the right, +: to the left

    :return: X: 3d tensor in the shape of (n, c, s), where:
    n is the number of 30-second windows in the time-series data set
    c is the number of signal sources, preliminarirly:
        0: ECG  1: PPG  2,3,4: x,y,z accelerometer  5: ENMO 6: z-angle  7: chest temperature 8: heart rate
    s the number of signal samples in each 30-second window (s = sample_rate * 30 seconds)
    """
    # Read raw signals from csv
    index, time, ecg, ppg, x_acc, y_acc, z_acc, enmo, z_angle, temp \
        = np.loadtxt(f"{DATA_DIR}/{subject_id}/ANNE.csv", delimiter=",", dtype="float32",
                     skiprows=1, unpack=True,
                     usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9))
    n = len(ecg) // (SAMP_RATE * WINDOW_LEN) * (SAMP_RATE * WINDOW_LEN)

    # Preliminary engineered features from ANNE.edf
    signals, signal_headers, header = highlevel.read_edf(
        f"{DATA_DIR}/{subject_id}/ANNE.edf")
    psg_samp_rate = int(signal_headers[7]["sample_rate"])

    # heart_rate = np.float32(np.repeat(signals[6], 25)[:len(ecg)])
    heart_rate = np.float32(signal.resample(signals[6], len(signals[6])*25)[:len(ecg)])

    # targets
    t = np.loadtxt(f"{DATA_DIR}/{subject_id}/sleepLabel.csv", dtype="int64", delimiter=",", usecols=1,
                   skiprows=1, unpack=True)

    # shift according to alignment
    if shift < 0:
        n_windows_left = ceil(abs(shift / WINDOW_LEN))
        n_samples_right = ceil(abs(shift * SAMP_RATE))
        n_samples_left = (n - n_samples_right) % (WINDOW_LEN * SAMP_RATE)
        t = t[n_windows_left:]

        # plot_ecg(ecg, subject_id, windows_dropped=n_windows_left)
        # plot_acc(ecg, subject_id, windows_dropped=n_windows_left)
    else:
        n_windows_right = ceil(abs(shift / WINDOW_LEN))
        n_samples_left = ceil(abs(shift * SAMP_RATE))
        n_samples_right = (n - n_samples_left) % (WINDOW_LEN * SAMP_RATE)
        t = t[:-n_windows_right]

    ecg = ecg[n_samples_left:-n_samples_right]
    ppg = ppg[n_samples_left:-n_samples_right]
    x_acc = x_acc[n_samples_left:-n_samples_right]
    y_acc = y_acc[n_samples_left:-n_samples_right]
    z_acc = z_acc[n_samples_left:-n_samples_right]
    enmo = enmo[n_samples_left:-n_samples_right]
    z_angle = z_angle[n_samples_left:-n_samples_right]
    temp = temp[n_samples_left:-n_samples_right]
    heart_rate = heart_rate[n_samples_left:-n_samples_right]

    n = n - n_samples_left - n_samples_right

    # Normalization and standardization
    ecg = (ecg - ecg.mean()) / np.sqrt(ecg.var())
    ppg = (ppg - ppg.mean()) / np.sqrt(ppg.var())
    temp = (temp - temp.mean()) / np.sqrt(temp.var())
    heart_rate = (heart_rate - heart_rate.mean()) / np.sqrt(heart_rate.var())
    z_angle = (z_angle - z_angle.mean()) / np.sqrt(z_angle.var())
    x_acc = (x_acc - x_acc.mean()) / np.sqrt(x_acc.var())
    y_acc = (y_acc - y_acc.mean()) / np.sqrt(y_acc.var())
    z_acc = (z_acc - z_acc.mean()) / np.sqrt(z_acc.var())


    # z_angle = (z_angle + 90) / 180
    # heart_rate = heart_rate / np.sqrt(heart_rate.var())

    # Features
    ecg = np.reshape(ecg[:n], (-1, 1, SAMP_RATE * WINDOW_LEN))
    ppg = np.reshape(ppg[:n], (-1, 1, SAMP_RATE * WINDOW_LEN))
    x_acc = np.reshape(x_acc[:n], (-1, 1, SAMP_RATE * WINDOW_LEN))
    y_acc = np.reshape(y_acc[:n], (-1, 1, SAMP_RATE * WINDOW_LEN))
    z_acc = np.reshape(z_acc[:n], (-1, 1, SAMP_RATE * WINDOW_LEN))
    enmo = np.reshape(enmo[:n], (-1, 1, SAMP_RATE * WINDOW_LEN))
    z_angle = np.reshape(z_angle[:n], (-1, 1, SAMP_RATE * WINDOW_LEN))
    temp = np.reshape(temp[:n], (-1, 1, SAMP_RATE * WINDOW_LEN))
    heart_rate = np.reshape(heart_rate[:n], (-1, 1, SAMP_RATE * WINDOW_LEN))

    (ecg_freq, ppg_freq, x_freq, y_freq, z_freq, zangle_freq), (hrv_s, ecg_s, ppg_s, z_angle_s, temp_s, enmo_s) = \
        get_f_s_features(ecg, ppg, enmo, z_angle, temp, heart_rate, x_acc, y_acc, z_acc)

    t = np.where((t == 2) | (t == 3), 1, t)
    t = np.where(t == 4, 2, t)

    t = np.reshape(t, (-1, len(t)))
    X = np.concatenate((ecg, ppg, enmo, z_angle, temp, heart_rate), axis=1)

    X_freq = np.concatenate((ecg_freq, ppg_freq, x_freq, y_freq, z_freq, zangle_freq), axis=1).astype('float32')

    X_scl = np.concatenate((hrv_s, ecg_s, ppg_s, z_angle_s, temp_s, enmo_s), axis=1).astype('float32')

    return X, X_freq, X_scl, t


def get_f_s_features(ecg, ppg, enmo, z_angle, temp, heart_rate, x_acc, y_acc, z_acc):
    ecg_freq = np.expand_dims(ecg_pwr_sptr(np.squeeze(ecg)), axis=1)
    ecg_freq = np.clip(ecg_freq, 0, np.mean(ecg_freq) + 4 * np.var(ecg_freq) ** 0.5) / np.var(ecg_freq) ** 0.5
    ppg_freq = np.expand_dims(ppg_pwr_sptr(np.squeeze(ppg)), axis=1)
    ppg_freq = np.clip(ppg_freq, 0, np.mean(ppg_freq) + 4 * np.var(ppg_freq) ** 0.5) / np.var(ppg_freq) ** 0.5
    x_freq = np.expand_dims(acc_pwr_sptr(np.squeeze(x_acc)), axis=1)
    x_freq = np.clip(x_freq, 0, np.mean(x_freq) + 4 * np.var(x_freq) ** 0.5) / (np.var(x_freq) ** 0.5)
    y_freq = np.expand_dims(acc_pwr_sptr(np.squeeze(y_acc)), axis=1)
    y_freq = np.clip(y_freq, 0, np.mean(y_freq) + 4 * np.var(y_freq) ** 0.5) / (np.var(y_freq) ** 0.5)
    z_freq = np.expand_dims(acc_pwr_sptr(np.squeeze(z_acc)), axis=1)
    z_freq = np.clip(z_freq, 0, np.mean(z_freq) + 4 * np.var(z_freq) ** 0.5) / (np.var(z_freq) ** 0.5)
    zangle_freq = np.expand_dims(z_angle_pwr_sptr(np.squeeze(z_angle)), axis=1)
    zangle_freq = np.clip(zangle_freq, 0, np.mean(zangle_freq) + 4 * np.var(zangle_freq) ** 0.5) / (
            np.var(z_freq) ** 0.5)

    # Scalar features
    hrv_s = hrv(ecg)
    ecg_s = ecg_pwr_sptr_scl(ecg)
    ppg_s = ppg_pwr_sptr_scl(ppg)
    z_angle_s = zangle_pwr_sptr_scl(z_angle)
    enmo_s = np.squeeze(enmo_scl(enmo))
    temp_s = np.squeeze(temp_scl(temp))
    hrv_s = np.expand_dims(hrv_s, axis=1)
    ecg_s = np.expand_dims(ecg_s, axis=1)
    ppg_s = np.expand_dims(ppg_s, axis=1)
    z_angle_s = np.expand_dims(z_angle_s, axis=1)
    enmo_s = np.expand_dims(enmo_s, axis=1)
    temp_s = np.expand_dims(temp_s, axis=1)

    return (ecg_freq, ppg_freq, x_freq, y_freq, z_freq, zangle_freq), (hrv_s, ecg_s, ppg_s, z_angle_s, temp_s, enmo_s)


def integrate_raw(subject_id, shift=0):
    """ Prepares the data tensor and target vector of subject with subject_id
    :param subject_id: the id of the subject for whom the data tensor and target will be prepared
    :param shift: shift of the ANNE signal in seconds, -: to the right, +: to the left

    :return: X: 3d tensor in the shape of (n, c, s), where:
    n is the number of 30-second windows in the time-series data set
    c is the number of signal sources, preliminarirly:
        0: ECG  1: PPG  2,3,4: x,y,z accelerometer  5: ENMO 6: z-angle  7: chest temperature 8: heart rate
    s the number of signal samples in each 30-second window (s = sample_rate * 30 seconds)
    """
    # Read raw signals from csv
    index, time, ecg, ppg, x_acc, y_acc, z_acc, enmo, z_angle, temp \
        = np.loadtxt(f"{DATA_DIR}/{subject_id}/ANNE.csv", delimiter=",", dtype="float32",
                     skiprows=1, unpack=True,
                     usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9))
    n = len(ecg) // (SAMP_RATE * WINDOW_LEN) * (SAMP_RATE * WINDOW_LEN)

    # Preliminary engineered features from ANNE.edf
    signals, signal_headers, header = highlevel.read_edf(
        f"{DATA_DIR}/{subject_id}/ANNE.edf")
    psg_samp_rate = int(signal_headers[7]["sample_rate"])

    # heart_rate = np.float32(np.repeat(signals[6], 25)[:len(ecg)])
    heart_rate = np.float32(signal.resample(signals[6], len(signals[6])*25)[:len(ecg)])

    # targets
    t = np.loadtxt(f"{DATA_DIR}/{subject_id}/sleepLabel.csv", dtype="int64", delimiter=",", usecols=1,
                   skiprows=1, unpack=True)

    # shift according to alignment
    if shift < 0:
        n_windows_left = ceil(abs(shift / WINDOW_LEN))
        n_samples_right = ceil(abs(shift * SAMP_RATE))
        n_samples_left = (n - n_samples_right) % (WINDOW_LEN * SAMP_RATE)
        t = t[n_windows_left:]

        # plot_ecg(ecg, subject_id, windows_dropped=n_windows_left)
        # plot_acc(ecg, subject_id, windows_dropped=n_windows_left)
    else:
        n_windows_right = ceil(abs(shift / WINDOW_LEN))
        n_samples_left = ceil(abs(shift * SAMP_RATE))
        n_samples_right = (n - n_samples_left) % (WINDOW_LEN * SAMP_RATE)
        t = t[:-n_windows_right]

    ecg = ecg[n_samples_left:-n_samples_right]
    ppg = ppg[n_samples_left:-n_samples_right]
    x_acc = x_acc[n_samples_left:-n_samples_right]
    y_acc = y_acc[n_samples_left:-n_samples_right]
    z_acc = z_acc[n_samples_left:-n_samples_right]
    enmo = enmo[n_samples_left:-n_samples_right]
    z_angle = z_angle[n_samples_left:-n_samples_right]
    temp = temp[n_samples_left:-n_samples_right]
    heart_rate = heart_rate[n_samples_left:-n_samples_right]

    n = n - n_samples_left - n_samples_right

    # Normalization and standardization
    # ecg = (ecg - ecg.mean()) / np.sqrt(ecg.var())
    # ppg = (ppg - ppg.mean()) / np.sqrt(ppg.var())
    # temp = (temp - temp.mean()) / np.sqrt(temp.var())
    # heart_rate = (heart_rate - heart_rate.mean()) / np.sqrt(heart_rate.var())
    # z_angle = (z_angle - z_angle.mean()) / np.sqrt(z_angle.var())
    # x_acc = (x_acc - x_acc.mean()) / np.sqrt(x_acc.var())
    # y_acc = (y_acc - y_acc.mean()) / np.sqrt(y_acc.var())
    # z_acc = (z_acc - z_acc.mean()) / np.sqrt(z_acc.var())

    # Features
    ecg = np.reshape(ecg[:n], (-1, 1, SAMP_RATE * WINDOW_LEN))
    ppg = np.reshape(ppg[:n], (-1, 1, SAMP_RATE * WINDOW_LEN))
    x_acc = np.reshape(x_acc[:n], (-1, 1, SAMP_RATE * WINDOW_LEN))
    y_acc = np.reshape(y_acc[:n], (-1, 1, SAMP_RATE * WINDOW_LEN))
    z_acc = np.reshape(z_acc[:n], (-1, 1, SAMP_RATE * WINDOW_LEN))
    enmo = np.reshape(enmo[:n], (-1, 1, SAMP_RATE * WINDOW_LEN))
    z_angle = np.reshape(z_angle[:n], (-1, 1, SAMP_RATE * WINDOW_LEN))
    temp = np.reshape(temp[:n], (-1, 1, SAMP_RATE * WINDOW_LEN))
    heart_rate = np.reshape(heart_rate[:n], (-1, 1, SAMP_RATE * WINDOW_LEN))

    t = np.where((t == 2) | (t == 3), 1, t)
    t = np.where(t == 4, 2, t)

    t = np.reshape(t, (-1, len(t)))
    X = np.concatenate((ecg, ppg, x_acc, y_acc, z_acc, enmo, z_angle, temp, heart_rate), axis=1)

    return X, t


def plot_ecg(anne, subject_id, start=0, stop=4200, anne_shift=0, windows_dropped=0):
    """
    Plot ANNE ecg signal against PSG ecg signals for visualizing the result of alignment
    """
    # Read PSG edf
    signals, signal_headers, header = highlevel.read_edf(
        f"{DATA_DIR}/{subject_id}/PSG.edf")
    psg_samp_rate = int(signal_headers[7]["sample_rate"])

    fig, axs = plt.subplots(2)
    axs[0].plot(np.linspace(start, stop, (stop - start) * SAMP_RATE),
                anne[(start + anne_shift) * SAMP_RATE:(stop + anne_shift) * SAMP_RATE])
    axs[0].title.set_text("ANNE")
    axs[1].plot(np.linspace(start, stop, (stop - start) * psg_samp_rate), signals[7][(
    start + windows_dropped * WINDOW_LEN) * psg_samp_rate:(stop + windows_dropped * WINDOW_LEN) * psg_samp_rate])
    axs[1].title.set_text("PSG")

    plt.show()

    pass


def plot_acc(anne, subject_id, start=0, stop=4200, anne_shift=0, windows_dropped=0):
    """
    Plot ANNE ecg signal against PSG ecg signals for visualizing the result of alignment
    """
    # Read PSG edf
    signals, signal_headers, header = highlevel.read_edf(
        f"{DATA_DIR}/{subject_id}/PSG.edf")
    psg_samp_rate = int(signal_headers[7]["sample_rate"])

    fig, axs = plt.subplots(2)
    axs[0].plot(np.linspace(start, stop, (stop - start) * SAMP_RATE),
                anne[(start + anne_shift) * SAMP_RATE:(stop + anne_shift) * SAMP_RATE])
    axs[0].title.set_text("ANNE")
    axs[1].plot(np.linspace(start, stop, (stop - start) * psg_samp_rate), signals[7][(
    start + windows_dropped * WINDOW_LEN) * psg_samp_rate:(stop + windows_dropped * WINDOW_LEN) * psg_samp_rate])
    axs[1].title.set_text("PSG")

    plt.show()

    pass


def plot_window(X, index=0):
    """
    Plot the index-th 30 second window in the data tensor X
    """
    fig, axs = plt.subplots(X.shape[1])
    for i in range(len(axs)):
        arr = X[index, i, :]
        axs[i].plot(arr)
    plt.show()


def augment_data(X):
    """Perform data augmentation given a time domain data set.
    Returns
    """
    X_freq, X_scl = None
    return X, X_freq, X_scl

if __name__ == "__main__":
    X, X_freq, t = integrate_data(132, -7.64)
    plot_window(X, index=0)
