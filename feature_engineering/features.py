import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import json

DATA_DIR = "/mnt/Common/Data"
SAMP_RATE = 25  # Hz
WINDOW_LEN = 30  # Seconds
PATIENT = 132
LAG = -7.64

N = SAMP_RATE * WINDOW_LEN
t_axis = np.arange(0, WINDOW_LEN, 1 / SAMP_RATE)
freq = np.fft.fftshift(np.fft.fftfreq(N, 1 / SAMP_RATE))[N // 2:]


# New features
def pwr_sptr(ppg):
    power = np.fft.fftshift(abs(np.fft.fft(ppg * 2 * np.hanning(N))))
    power = power[:, :N // 2] * 1 / SAMP_RATE
    return power


def ppg_entropy(ppg):
    entropy = stats.entropy((np.squeeze(ppg) - np.min(ppg)), axis=-1)
    return entropy


def ppg_pwr_sptr_scl(ppg):
    return stats.skew(pwr_sptr(np.squeeze(ppg)), axis=-1)


def ppg_pwr_sptr_entp(ppg):
    return stats.entropy(pwr_sptr(np.squeeze(ppg)), axis=-1)


def hr_sclr(hr):
    scl = np.max(np.squeeze(hr), axis=-1) - np.min(np.squeeze(hr), axis=-1)
    return scl


def enmo_scl(enmo):
    return abs(stats.kurtosis((np.squeeze(enmo)), axis=-1)) ** 0.25


def zangle_scl(zangle):
    return stats.entropy(pwr_sptr(np.squeeze(zangle)), axis=-1)


def sqi_avg(sqi):
    return np.mean(np.squeeze(sqi), axis=-1)

def system(v, delta):
    return 2 * delta * np.sin(2 * np.pi * v * delta) / (2 * np.pi * v * delta)
