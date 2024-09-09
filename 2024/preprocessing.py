"""
Code for integrating raw data into appropriate formats for later use
@author: Andrew H. Zhang
"""
from mne.io import read_raw_edf
from pathlib import Path
from datetime import datetime
import multiprocessing as mp
import numpy as np
import pandas as pd
import os
import scipy
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Union

ML_SAMPLING_RATE = 25  # Hz
EDF_SAMPLING_RATE = 100  # Hz
WINDOW_LENGTH = 30  # Seconds

RUN_LOCALLY = os.name == 'nt'
if RUN_LOCALLY:
    DATA_PATH = "../../ApneaProject2024/ANNE-PSG240421 toy data"
    METADATA_PATH = DATA_PATH
    OUTPUT_PATH = "../ANNE-PSG240421 toy features"
else:
    DATA_PATH = '/scratch/a/alim/alim/overnight_validation/ANNE-PSG240421'
    METADATA_PATH = '/home/a/alim/yrichard'
    OUTPUT_PATH = '/scratch/a/alim/yrichard/anne-psg-dl-features'


####################################################################################################
# Helper functions
####################################################################################################
def read_files_in_path(path: str) -> list[str]:
    """
    Read dataset files (location specified by path), returning the i-th file's path as the i-th
    element of the return list.

    The data is structured as many folders, where each folder contains a single EDF file of the same
    name as the folder plus a suffix '-features.edf' or '-score.edf'
    """
    paths = []

    for entry in Path(path).iterdir():
        if not entry.is_dir():
            continue

        for sub_entry in entry.iterdir():
            paths.append(str(sub_entry))

    return paths


def edf_to_numpy(path: str, kept_columns: list[str],
                 same_sample_rate: bool = True) -> Union[np.ndarray, list[np.ndarray]]:
    """
    Convert the EDF file at path into numpy arrays.
    If same_sample_rate, the output is an array (D, N); else, it's a list of variable-sized arrays.
    If no columns are specified, all of them are returned.
    """
    # Read single columns to stop resampling artifacts when different sample rate signals are read
    if len(kept_columns) > 0:
        data = []
        for column in kept_columns:
            try:
                data.append(read_raw_edf(path, verbose=False, include=[column])[:][0][0])
            except ValueError:
                raise AssertionError(f"{os.path.basename(path)} does not contain label {column}")
    else:
        data = [read_raw_edf(path, verbose=False, include=[column])[:][0][0] for column in
                read_raw_edf(path, verbose=False).ch_names]

    if not same_sample_rate:
        return data

    # Padding low sample-rate data
    max_sample_len = max(len(x) for x in data)
    upsample_factor = [max_sample_len // len(x) for x in data]
    return np.stack([np.repeat(data[i], upsample_factor[i]) for i in range(len(data))])


def extract_metadata(metadata_path: str, id: str) -> tuple[float, float]:
    metadata = pd.read_csv(f"{metadata_path}/ANNE-PSG_metadata.csv")

    i = np.where(metadata["file"] == id)[0]
    if len(i) == 0:
        print(f"[WARNING] {id}'s metadata could not be found")
        age = -1
        sex = -1
    else:
        age = metadata['age'][i].values[0]
        sex = metadata['sex'][i].values[0]
        age = -1 if np.isnan(age) else age / 100
        sex = -1 if not isinstance(sex, str) else int(sex == 'Male')
    return age, sex


def sanity_check_labels(path: str, sleepstage: np.ndarray) -> bool:
    path = os.path.basename(path)
    success = True

    if not all(x in {0, 1, 2, 3, 4, 5} for x in np.unique(sleepstage)):
        print(f"[ERROR] {path} has unknown sleepstage labels {np.unique(sleepstage)}")
        success = False

    if np.all(sleepstage > 0):
        print(f"[ERROR] {path} has no awake labels")
        success = False
    elif np.all(sleepstage == 0):
        print(f"[WARNING] {path} has no asleep labels")
        success = False

    return success


def downsample_nearest(signal: np.ndarray, new_length: int, reduction: str = 'max',
                       keepdims: bool = True) -> np.ndarray:
    """
    Downsample a 1D signal to a new length using a given reduction.
    Callable should be lambda x: np.__(x, axis=0, keepdims=True)
    """
    if reduction == 'max':
        reduction = lambda x: np.max(x, axis=0, keepdims=keepdims)
    elif reduction == 'mean':
        reduction = lambda x: np.mean(x, axis=0, keepdims=keepdims)
    else:
        raise ValueError()

    diff = len(signal) / new_length
    i = np.floor(np.arange(0, len(signal), diff))
    if len(i) < new_length:
        i = np.append(i, np.ones(new_length - len(i)) * i[-1])
    elif len(i) > new_length:
        i = i[:new_length]
    return reduction(
        signal[np.clip(np.add.outer(np.arange(np.floor(diff)), i), 0, len(signal) - 1).astype(int)])


####################################################################################################
# Features
####################################################################################################
def get_sleepstage_transitions(t: np.ndarray) -> np.ndarray:
    t_diff = np.append(np.ediff1d(t), -t[-1])
    return (t_diff != 0).astype(float)


def power_spectrum(data: np.ndarray, sample_rate: int, window_length: int) -> np.ndarray:
    power = np.fft.fftshift(
        abs(np.fft.fft(data * 2 * np.hanning(sample_rate * window_length), axis=-1)), axes=-1)
    return power[..., :sample_rate * window_length // 2] / sample_rate


def ppg_scalar_features(ppg: np.ndarray, ppg_freq: np.ndarray) -> np.ndarray:
    return np.stack([
        scipy.stats.entropy(ppg - np.minimum(0, np.min(ppg, axis=-1, keepdims=True)), axis=-1),
        scipy.stats.skew(ppg_freq, axis=-1),
        scipy.stats.entropy(ppg_freq - np.minimum(0, np.min(ppg_freq, axis=-1, keepdims=True)), axis=-1)
    ])


def hr_scale(hr: np.ndarray) -> np.ndarray:
    return np.max(hr - np.min(hr, axis=-1, keepdims=True), axis=-1)


def enmo_kurtosis(enmo: np.ndarray) -> np.ndarray:
    return abs(scipy.stats.kurtosis(enmo, nan_policy='omit', axis=-1)) ** 0.25


def z_angle_power_spectrum_entropy(z_angle: np.ndarray, sample_rate: int,
                                   window_length: int) -> np.ndarray:
    return scipy.stats.entropy(power_spectrum(z_angle, sample_rate, window_length), axis=-1)


####################################################################################################
# Preprocessing
####################################################################################################
def preprocess(path: str, metadata_path: str = METADATA_PATH,
               inference: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Preprocess an EDF file at path.

    For path A-score.edf, metadata_path should point to a CSV with columns ['age', 'sex'] and row A.
    metadata_path can be set to '', in which case the age/sex will be set to -1 (unknown).
    """
    hr_ecg = 'HR.ECG' if inference else 'HR_ECG'
    hr_ppg = 'HR.PPG' if inference else 'HR_PPG'
    hr_combi = 'HR.combi' if inference else 'HR_combi'
    features = ['ecgProcessed', 'ppgFiltered', 'PPGamp', 'x', 'y', 'z', 'chestTemp', 'limbTemp',
                hr_ecg, hr_ppg, hr_combi, 'PAT', 'theta', 'phi', 'ecgSQI', 'ppgSQI']
    if not inference:
        features.append('sleep.stage')

    # Load EDF
    try:
        data = edf_to_numpy(path, features, same_sample_rate=True)  # (D, N)
        ind = lambda x: features.index(x)
    except Exception as e:
        print(f"[ERROR] {e}")
        return np.array([]), np.array([]), np.array([])

    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        print(f"[ERROR] Detected NaN/inf values in {os.path.basename(path)}")
        return np.array([]), np.array([]), np.array([])

    # Detrend
    detrend = [ind('ecgProcessed'), ind('ppgFiltered'), ind('x'), ind('y'), ind('z')]
    data[detrend] -= np.mean(data[detrend], axis=1, keepdims=True)

    # Resample
    n = data.shape[1] * ML_SAMPLING_RATE // EDF_SAMPLING_RATE
    if inference:
        data = [downsample_nearest(data[i], n, 'mean', True) for i in range(data.shape[0])]
    else:
        data = [downsample_nearest(data[i], n, 'mean', True) for i in range(data.shape[0] - 1)] \
               + downsample_nearest(data[-1], n, 'max', True)
    data = np.concatenate(data, axis=0)

    # Label preprocessing
    if not inference:
        sleepstage = np.round(data[ind('sleep.stage')]).astype(int)
        if not sanity_check_labels(path, sleepstage):
            return np.array([]), np.array([]), np.array([])

        sleepstage[(sleepstage > 1) & (sleepstage < 5)] = 1
        sleepstage[sleepstage == 5] = 2

        sleepstage_transitions = get_sleepstage_transitions(sleepstage)
        data[ind('sleep.stage')] = sleepstage
        data = np.concatenate([data, np.expand_dims(sleepstage_transitions, 0)], axis=0)

    # Divide into windows
    window_size = ML_SAMPLING_RATE * WINDOW_LENGTH
    num_points = window_size * (data.shape[1] // window_size)
    data = data[:, -num_points:]
    data = data.reshape(data.shape[0], data.shape[1] // window_size, window_size)  # (D, N // K, K)

    # New features
    x, y, z = data[ind('x')], data[ind('y')], data[ind('z')]
    z_angle = np.arctan(z / ((x ** 2 + y ** 2) ** 0.5)) / (np.pi / 180)
    enmo = (x ** 2 + y ** 2 + z ** 2) ** 0.5 - 1
    temp_diff = data[ind('chestTemp')] - data[ind('limbTemp')]

    # Fourier transform
    frequency_data = np.concatenate([
        data[[ind('ecgProcessed'), ind('ppgFiltered')]],
        x[None],
        y[None],
        z[None],
        z_angle[None],
        # data[np.array([ind('HR_ECG'), ind('HR_PPG')])]
    ], axis=0)
    frequency_features = power_spectrum(frequency_data, ML_SAMPLING_RATE, WINDOW_LENGTH)
    mu = np.mean(frequency_features, axis=(1, 2), keepdims=True)
    std = np.std(frequency_features, axis=(1, 2), keepdims=True)
    frequency_features = np.clip(frequency_features, 0, mu + 4 * std) / std
    frequency_features = np.log(frequency_features)  # (D2, N // K, K // 2)

    # Metadata
    filename = os.path.basename(path)
    filename = filename[:filename.rfind('-')]
    if metadata_path == '':
        age, sex = -1, -1
    else:
        age, sex = extract_metadata(metadata_path, filename)
    age = np.full(data.shape[1], age, dtype=float)
    sex = np.full(data.shape[1], sex, dtype=float)

    # Scalars
    scalar_features = np.stack([
        age,
        sex,
        np.mean(data[ind('ppgSQI')], axis=-1),
        np.mean(data[ind('ecgSQI')], axis=-1),
        *ppg_scalar_features(data[ind('ppgFiltered')], frequency_features[ind('ppgFiltered')]),
        hr_scale(data[ind(hr_combi)]),
        enmo_kurtosis(enmo),
        z_angle_power_spectrum_entropy(z_angle, ML_SAMPLING_RATE, WINDOW_LENGTH),
    ])  # (D3, N // K)
    bad = np.isnan(scalar_features) | np.isinf(scalar_features)
    if np.any(bad):
        bad_dims = np.where(np.any(bad, axis=1))[0]
        print(f'[WARNING] {filename} has NaN/inf values in dimensions {bad_dims}')
        scalar_features[bad] = 0

    # Regular data
    data = np.concatenate([
        data[[ind(hr_combi), ind(hr_ppg), ind(hr_ecg), ind('PAT')]],
        enmo[None],
        z_angle[None],
        temp_diff[None],
        data[[ind('theta'), ind('phi')] + ([] if inference else [ind('sleep.stage'), -1])]
    ], axis=0)  # (D1, N // K, K)

    # Reshape
    data = np.transpose(data, axes=(1, 0, 2))                               # (N // K, D1, K)
    frequency_features = np.transpose(frequency_features, axes=(1, 0, 2))   # (N // K, D2, K // 2)
    scalar_features = np.transpose(scalar_features, axes=(1, 0))            # (N // K, D3)

    if not inference:
        np.save(f'{OUTPUT_PATH}/{filename}_data_scalar.npy', scalar_features)
        np.save(f'{OUTPUT_PATH}/{filename}_data.npy', data)
        np.save(f'{OUTPUT_PATH}/{filename}_data_frequency.npy', frequency_features)
    return data, frequency_features, scalar_features


if __name__ == "__main__":
    t1 = datetime.now()
    print(f"Started job at {t1}")

    paths = read_files_in_path(DATA_PATH)
    print(f"{len(paths)} files to process")

    if RUN_LOCALLY:
        # for path in tqdm(paths):
        #     preprocess(path)
        # preprocess(paths[0], inference=True)
        pass
    else:
        cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', default=1))
        pool = mp.Pool(processes=cpus)
        data = pool.map(preprocess, paths)
        pool.close()
        pool.join()

    t2 = datetime.now()
    print(f"Finished job at {t2} with job duration {t2 - t1}")
