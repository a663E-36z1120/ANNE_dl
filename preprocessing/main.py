# Code for integrating raw data into appropriate formats for later use
# @author: Andrew H. Zhang

from matplotlib import pyplot as plt
from scipy import signal
from feature_engineering.features import ppg_entropy, ppg_pwr_sptr_scl, ppg_pwr_sptr_entp, hr_sclr, enmo_scl, zangle_scl
import numpy as np
import mne
import os
import json
import gc
import pandas as pd
import math

DATA_DIR = "/media/a663e-36z/Common/Data/ANNE-data-expanded"
ML_SAMP_RATE = 25  # Hz
EDF_SAMP_RATE = 100  # Hz
WINDOW_LEN = 30  # Seconds

t_axis = np.arange(0, WINDOW_LEN, 1 / ML_SAMP_RATE)
N = ML_SAMP_RATE * WINDOW_LEN

meta_file = "/media/a663e-36z/Common/Data/ANNE-data-expanded/ANNE-PSG_metadata.csv"
METADATA = pd.read_csv(meta_file)

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

    if t[0] > 4:
        t[0] = 0

    mask = (t > 4)
    idx = np.where(~mask, np.arange(mask.shape[0]), 0)
    np.maximum.accumulate(idx, axis=0, out=idx)
    t[mask] = t[idx[mask]]

    t = np.where(t == 4, 2, t)
    return t




def process_time_series(series, start_index, end_index):
    series = (series - series.mean()) / np.sqrt(series.var())
    data_matrix = np.reshape(series[start_index:end_index], (-1, 1, EDF_SAMP_RATE * WINDOW_LEN))

    return data_matrix

def detrend(x, y, degree=5):
    detrended = np.empty_like(y)

    for i in range(len(y)):
        params = np.polyfit(x, y[i], degree)
        line = 0
        for d in range(degree):
            line += x ** (degree - d) * params[d]

        line += params[-1]
        detrended[i] = y[i] - line
    return detrended

def pwr_sptr(signals):
    signals = detrend(t_axis, signals)

    power = np.fft.fftshift(abs(np.fft.fft(signals * 2 * np.hanning(N))))
    power = power[:, N // 2:N // 4 * 3] * 1 / ML_SAMP_RATE
    return power


def get_frequency_features(signals):
    signals = signal.resample(signals, ML_SAMP_RATE * WINDOW_LEN, axis=2)
    freq = np.expand_dims(pwr_sptr(np.squeeze(signals)), axis=1)
    freq = np.clip(freq, 0, np.mean(freq) + 4 * np.var(freq) ** 0.5) / (np.var(freq) ** 0.5)
    # plt.plot(freq[100,0,:])
    # plt.show()
    return freq


def get_scalar_features():
    pass


def main(path, inference=False, feature_engineering=False, no_ppg=False, save_path=None):
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

    if no_ppg:
        X = np.concatenate((hr, enmo, z_angle, rr, temp_diff), axis=1)
    else:
        X = np.concatenate((hr, pat, enmo, z_angle, rr, temp_diff), axis=1)


    X = signal.resample(X, ML_SAMP_RATE * WINDOW_LEN, axis=2).astype('float32')

    x_acc, y_acc, z_acc = signals_map[7], signals_map[8], signals_map[9]

    # Frequency domain features
    ecg_freq, ppg_freq, x_freq, y_freq, z_freq, z_angle_freq = \
        get_frequency_features(ecg), get_frequency_features(ppg), get_frequency_features(x_acc), get_frequency_features(
            y_acc), get_frequency_features(z_acc), get_frequency_features(z_angle)

    if no_ppg:
        X_freq = np.concatenate((ecg_freq, x_freq, y_freq, z_freq), axis=1).astype('float32')
    else:
        X_freq = np.concatenate((ecg_freq, ppg_freq, x_freq, y_freq, z_freq), axis=1).astype('float32')


    # Scalar domain features
    ppg_resamp = signal.resample(ppg, ML_SAMP_RATE * WINDOW_LEN, axis=2).astype('float32')
    ppg_entpy = ppg_entropy(ppg_resamp)
    ppg_sptr_skw = ppg_pwr_sptr_scl(ppg_resamp)
    ppg_sptr_entp = ppg_pwr_sptr_entp(ppg_resamp)
    hr_resamp = signal.resample(ppg, ML_SAMP_RATE * WINDOW_LEN, axis=2).astype('float32')
    hr_diff = hr_sclr(hr_resamp)
    enmo_resamp = signal.resample(ppg, ML_SAMP_RATE * WINDOW_LEN, axis=2).astype('float32')
    enmo_kurt = enmo_scl(enmo_resamp)
    zangle_resamp = signal.resample(z_angle, ML_SAMP_RATE * WINDOW_LEN, axis=2).astype('float32')
    zangle_sptr_entp = zangle_scl(zangle_resamp)

    # Engineered Scalers
    ppg_entpy = np.expand_dims(ppg_entpy, axis=1) / np.mean(ppg_entpy)
    ppg_sptr_skw = np.expand_dims(ppg_sptr_skw, axis=1) / np.mean(ppg_sptr_skw)
    ppg_sptr_entp = np.expand_dims(ppg_sptr_entp, axis=1) / np.mean(ppg_sptr_entp)
    hr_diff = np.expand_dims(hr_diff, axis=1) / np.mean(hr_diff)
    enmo_kurt = np.expand_dims(enmo_kurt, axis=1) / np.mean(enmo_kurt)
    zangle_sptr_entp = np.expand_dims(zangle_sptr_entp, axis=1) / np.mean(zangle_sptr_entp)
    # Meta Scalers
    age, sex = extract_metadata(path)
    age = np.full_like(ppg_entpy, age)
    sex = np.full_like(ppg_entpy, sex)

    if no_ppg:
        X_scl = np.concatenate((enmo_kurt, zangle_sptr_entp, hr_diff, age, sex),
                               axis=1).astype('float32')
    else:
        X_scl = np.concatenate((ppg_entpy, ppg_sptr_skw, ppg_sptr_entp, enmo_kurt, zangle_sptr_entp, hr_diff, age, sex), axis=1).astype('float32')



    if feature_engineering:
        X_raw = np.concatenate((ecg, ppg, enmo, z_angle, spo2, hr, rr), axis=1).astype('float32')
        X_raw = signal.resample(X_raw, ML_SAMP_RATE * WINDOW_LEN, axis=2).astype('float32')
        gc.collect()
        return X_raw, t

    gc.collect()

    # Save the data if save_path is provided
    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump((X, X_freq, X_scl, t), f)
        print(f"Saved to {save_path}.")

    return X, X_freq, X_scl, t


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


def clean_filename(input_string, prefix=DATA_DIR+"/", suffix='-annotated.edf'):
    # Check if the input string starts with the specified prefix
    if input_string.startswith(prefix):
        # Remove the prefix
        input_string = input_string[len(prefix):]

    # Check if the input string ends with the specified suffix
    if input_string.endswith(suffix):
        # Remove the suffix
        input_string = input_string[:-len(suffix)]

    return input_string

def extract_metadata(path):
    filename = clean_filename(path)
    print(filename)
    row_idx = (METADATA["filename"] == filename)
    row = METADATA[row_idx]
    age = row.iloc[0]["age"]
    sex = row.iloc[0]["sex"]

    print(f"age: {age}; sex: {sex}")
    sex = str(sex)
    if sex.lower() == "male":
        sex = 1
    elif sex.lower() == "female":
        sex = 0
    else:
        sex = 0.5

    if not math.isnan(age):
        age = age / 100
    else:
        age = 0.5

    return age, sex



if __name__ == "__main__":
    X, X_freq, X_scl, t = main("/media/a663e-36z/Common/Data/ANNE-data-expanded/23-09-26-19_33_55.C4408.L4087.674-annotated.edf")
    fig, axs = plt.subplots(X_freq.shape[1], sharex=True)
    print(X.shape)
    for i in range(X_freq.shape[1]):
        axs[i].plot(X_freq[989,i,:])

    plt.show()




    # paths = get_edf_files("/media/a663e-36z/Common/Data/ANNE-data-expanded/")
    # targets = []
    # for path in paths:
    #     data = mne.io.read_raw_edf(path)
    #     raw_data = data.get_data()
    #
    #     # create target vector
    #     target_array = raw_data[27]
    #     plt.plot(target_array)
    #     plt.show()
    #
    #     start_index, end_index = get_valid_indices(target_array, EDF_SAMP_RATE)
    #     t = process_target(target_array, start_index, end_index)
    #
    #     targets.append(t)
    #
    # result = count_elements(targets)
    # wake_count, nrem_count, rem_count = result[0], result[1], result[2]
    #
    # # # t = np.reshape(t, (-1, len(t)))
    # # plt.plot(t)
    # # plt.savefig(f"{path[:-4]}.png")
    # # plt.clf()
    #
    # total = wake_count + rem_count + nrem_count
    # print(total)
    # print(f"wake: {wake_count / total}")
    # print(f"nrem: {nrem_count / total}")
    # print(f"rem: {rem_count / total}")
