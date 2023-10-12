# Code for integrating raw data into appropriate formats for later use
# @author: Andrew H. Zhang

from matplotlib import pyplot as plt
from scipy import signal
# from feature_engineering.features import *
import numpy as np
import mne
import os
import json
import gc

DATA_DIR = "/mnt/Common/Data"
ML_SAMP_RATE = 25  # Hz
EDF_SAMP_RATE = 100  # Hz
WINDOW_LEN = 30  # Seconds


def plot_window(X, index=0):
    """
    Plot the index-th 30 second window in the data tensor X
    """
    fig, axs = plt.subplots(X.shape[1])
    for i in range(len(axs)):
        arr = X[index, i, :]
        axs[i].plot(arr)
    plt.show()


def get_valid_indices(target_array, samp_rate):
    first = 0
    for j in range(len(target_array)):
        if target_array[j] != target_array[j - 1] and target_array[j] < 6:
            first = j
            while first > samp_rate * WINDOW_LEN:
                first -= samp_rate * WINDOW_LEN
            break

    tail = (len(target_array) - first) % (WINDOW_LEN * samp_rate)
    return first, len(target_array) - tail


def process_target(target_array, start_index, end_index):
    target_array = target_array[start_index:end_index]
    target_matrix = np.reshape(target_array, (-1, EDF_SAMP_RATE * WINDOW_LEN))
    t = target_matrix[:, 0].astype(int)

    t = np.where((t == 2) | (t == 3), 1, t)
    t = np.where(t == 4, 2, t)
    t = np.where(t > 4, 2, t)
    # temporarily classify the unknown class as NREM

    if t[0] == 9:
        t[0] = 1

    mask = (t == 9)
    idx = np.where(~mask, np.arange(mask.shape[0]), 0)
    np.maximum.accumulate(idx, axis=0, out=idx)
    t[mask] = t[idx[mask]]

    #t = np.where(t == 9, 1, t)
    t = np.where(t > 4, 2, t)

    return t


def process_time_series(series, start_index, end_index):
    series = (series - series.mean()) / np.sqrt(series.var())
    data_matrix = np.reshape(series[start_index:end_index], (-1, 1, EDF_SAMP_RATE * WINDOW_LEN))

    return data_matrix


def pwr_sptr(signals):
    N = ML_SAMP_RATE * WINDOW_LEN
    power = np.fft.fftshift(abs(np.fft.fft(signals * 2 * np.hanning(N))))
    power = power[:, N // 2:] * 1 / ML_SAMP_RATE
    return power


def get_frequency_features(signals):
    signals = signal.resample(signals, ML_SAMP_RATE * WINDOW_LEN, axis=2)
    freq = np.expand_dims(pwr_sptr(np.squeeze(signals)), axis=1)
    freq = np.clip(freq, 0, np.mean(freq) + 4 * np.var(freq) ** 0.5) / (np.var(freq) ** 0.5)
    return freq


def get_scalar_features():
    pass


def main(path, inference=False, feature_engineering=False):
    data = mne.io.read_raw_edf(path)
    raw_data = data.get_data()

    if inference:
        target_array = np.zeros_like(raw_data[0])
        start_index = 0
        end_index = len(target_array) - (len(target_array) % (WINDOW_LEN * EDF_SAMP_RATE))
    else:
        # create target vector
        target_array = raw_data[27]
        start_index, end_index = get_valid_indices(target_array, EDF_SAMP_RATE)

    t = process_target(target_array, start_index, end_index)
    t = np.reshape(t, (-1, len(t)))

    # create feature matrices

    ecg, ppg, x_acc, y_acc, z_acc, chest_temp, limb_temp, pat, hr, spo2, rr = \
        None, None, raw_data[7], raw_data[8], raw_data[9], raw_data[10], raw_data[11], None, None, None, None

    enmo = np.sqrt(x_acc ** 2 + y_acc ** 2 + z_acc ** 2) - 1
    z_angle = (np.arctan(z_acc / (np.sqrt(np.power(x_acc, 2) + np.power(y_acc, 2))))) / (np.pi / 180)
    z_angle = np.float32(z_angle)
    temp_diff = chest_temp - limb_temp

    signals_map = {
        2: ecg, 5: ppg, 7: x_acc, 8: y_acc, 9: z_acc, 10: chest_temp, 11: limb_temp,
        17: pat, 21: hr, 22: spo2, 23: rr,
        -1: enmo, -2: z_angle, -3: temp_diff
    }
    #  02: ecg processed -
    #  05: ppg processed -
    #  07 08 09: x, y, z acceleration -
    #  10: chest temp -
    #  11: limb temp -
    #  17: PAT detrend -
    #  21: HR -
    #  22: SpO2 -
    #  23: RR -

    for index in signals_map.keys():
        if index >= 0:
            signals = raw_data[index]
        else:
            signals = signals_map[index]
        # signals = signal.resample(signals, len(signals)//(EDF_SAMP_RATE // ML_SAMP_RATE))
        # start_index, end_index = get_valid_indices(signals, ML_SAMP_RATE)
        signals = process_time_series(signals, start_index, end_index)
        signals_map[index] = signals

    ecg = signals_map[2]
    ppg = signals_map[5]
    x_acc, y_acc, z_acc = signals_map[7], signals_map[8], signals_map[9]
    # chest_temp, limb_temp = signals_map[10], signals_map[11]
    enmo = signals_map[-1]
    z_angle = signals_map[-2]
    temp_diff = signals_map[-3]
    pat = signals_map[17]
    hr = signals_map[21]
    rr = signals_map[23]
    spo2 = signals_map[22]

    X = np.concatenate((hr, pat, enmo, z_angle, rr, temp_diff), axis=1)
    X = signal.resample(X, ML_SAMP_RATE * WINDOW_LEN, axis=2).astype('float32')

    x_acc, y_acc, z_acc = signals_map[7], signals_map[8], signals_map[9]

    ecg_freq, ppg_freq, x_freq, y_freq, z_freq, z_angle_freq = \
        get_frequency_features(ecg), get_frequency_features(ppg), get_frequency_features(x_acc), get_frequency_features(
            y_acc), get_frequency_features(z_acc), get_frequency_features(z_angle)

    X_freq = np.concatenate((ecg_freq, ppg_freq, x_freq, y_freq, z_freq), axis=1).astype('float32')

    if feature_engineering:
        X_raw = np.concatenate((ecg, ppg, enmo, spo2, hr, rr), axis=1).astype('float32')
        X_raw = signal.resample(X_raw, ML_SAMP_RATE * WINDOW_LEN, axis=2).astype('float32')
        gc.collect()
        return X_raw, t

    gc.collect()
    return X, X_freq, np.zeros(shape=(len(X), 1)), t


def get_edf_files(directory):
    edf_files = []

    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(".edf"):
                edf_files.append(os.path.join(root, file))

    return edf_files


def count_elements(arrays_list):
    counts = {
        0: 0,
        1: 0,
        2: 0
    }

    for arr in arrays_list:
        unique, counts_per_arr = np.unique(arr, return_counts=True)
        for element, count in zip(unique, counts_per_arr):
            counts[element] += count

    return counts


def read_strings_from_json(filename):
    with open(filename, "r") as json_file:
        data = json.load(json_file)
        if "strings" in data:
            return data["strings"]
        else:
            return []


if __name__ == "__main__":
    main("/mnt/Common/Downloads/23-03-22-21_41_32.C4359.L3786.570-annotated.edf")
    # edf_files = get_edf_files("/mnt/Common/data")
    # print(edf_files)
    # print(len(edf_files))
    # sample = random.sample(edf_files, 30)
    # print(len(sample))

    paths = get_edf_files("/mnt/Common/data")

    targets = []
    for path in paths:
        data = mne.io.read_raw_edf(path)
        raw_data = data.get_data()

        # create target vector
        target_array = raw_data[27]
        # plt.plot(target_array)
        plt.show()

        start_index, end_index = get_valid_indices(target_array, EDF_SAMP_RATE)
        t = process_target(target_array, start_index, end_index)

        targets.append(t)

    result = count_elements(targets)
    wake_count, nrem_count, rem_count = result[0], result[1], result[2]

    # # t = np.reshape(t, (-1, len(t)))
    # plt.plot(t)
    # plt.savefig(f"{path[:-4]}.png")
    # plt.clf()

    total = wake_count + rem_count + nrem_count
    print(total)
    print(f"wake: {wake_count / total}")
    print(f"nrem: {nrem_count / total}")
    print(f"rem: {rem_count / total}")
