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


def ecg_pwr_sptr(ecg):
    power = np.fft.fftshift(abs(np.fft.fft(ecg * 2 * np.hanning(N))))
    power = power[:, N // 2:] * 1 / SAMP_RATE
    return power


def ecg_pwr_sptr_scl(ecg):
    return np.var(ecg_pwr_sptr(ecg))


def hrv(ecg):
    power = ecg_pwr_sptr(ecg) ** 2
    lf_range = np.where((0.04 <= freq) & (freq >= 0.15), power, 0)
    lf = np.sum(lf_range)

    hf_range = np.where((0.15 <= freq) & (freq >= 0.4), power, 0)
    hf = np.sum(hf_range)

    return lf / hf


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


def ppg_pwr_sptr(ppg):
    ppg = detrend(t_axis, ppg)
    power = np.fft.fftshift(abs(np.fft.fft(ppg * 2 * np.hanning(N))))
    power = power[:, N // 2:] * 1 / SAMP_RATE
    return power


def ppg_pwr_sptr_scl(ppg):
    return stats.skew(ppg_pwr_sptr(ppg))


def hr_sclr(hr):
    return stats.skew(hr)


def acc_pwr_sptr(acc):
    acc = detrend(t_axis, acc, degree=1)
    power = np.fft.fftshift(abs(np.fft.fft(acc * 2 * np.hanning(N))) ** 0.25)
    power = power[:, N // 2:] * 1 / SAMP_RATE
    return power


def acc_pwr_sptr_scl(acc):
    return np.var(acc_pwr_sptr(acc))


def z_angle_pwr_sptr(z_angle):
    z_angle = detrend(t_axis, z_angle, degree=1)
    power = np.fft.fftshift(abs(np.fft.fft(z_angle * 2 * np.hanning(N))) ** 0.25)
    power = power[:, N // 2:] * 1 / SAMP_RATE
    return power


def zangle_pwr_sptr_scl(zangle):
    return stats.skew(z_angle_pwr_sptr(zangle))


def enmo_scl(enmo):
    return stats.kurtosis(enmo)


def temp_scl(temp):
    return np.mean(temp)
