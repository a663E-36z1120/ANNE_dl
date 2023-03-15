import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, lfilter
from pyedflib import highlevel
from scipy import signal
import os
import json

DATA_DIR = "/mnt/Common/Data"
sample_rate = 25


def butter_lowpass_filter(data, cutoff, fs, order):
    normal_cutoff = cutoff
    b, a = butter(order, normal_cutoff, fs=fs, btype='lowpass', output="ba")
    y = filtfilt(b, a, data)
    return y

def butter_bandpass(lowcut, highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype='band')

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def align(patientid, ANNE, PSG):
    # fs = sample_rate  # sample rate, Hz
    # order = 2  # sin wave can be approx represented as quadratic

    # ANNE_filt = butter_bandpass_filter(ANNE, 0.6, 2, fs, order)
    PSG_filt = butter_bandpass_filter(PSG, 0.05, 1, fs, order)

    # ANNE_filt = ANNE
    PSG_filt = PSG

    corr = signal.correlate(ANNE_filt, PSG_filt)
    lags = signal.correlation_lags(len(PSG_filt), len(ANNE_filt))
    corr /= np.max(corr)
    len_corr = len(corr)

    center = np.where(lags == 0)[0][0]
    radius = 30

    corr_bounded = corr[center - sample_rate * radius:center + sample_rate * radius]
    lags_bounded = lags[center - sample_rate * radius:center + sample_rate * radius]

    sample_lag = lags_bounded[np.argmax(corr_bounded)]

    # Generate some plots around artifacts to assess quality of alignment

    wr = 15  # window radius around each artifact to generate the plot for

    min_len = min(len(PSG_filt), len(ANNE_filt))

    diff = np.abs(PSG_filt[:min_len] - ANNE_filt[:min_len])

    artifact0 = np.argmax(diff[:min_len // 5])
    artifact1 = min_len // 5 + np.argmax(diff[min_len // 5: min_len // 5 * 2])
    artifact2 = min_len // 5 * 2 + np.argmax(diff[min_len // 5 * 2: min_len // 5 * 3])
    artifact3 = min_len // 5 * 3 + np.argmax(diff[min_len // 5 * 3: min_len // 5 * 4])
    artifact4 = min_len // 5 * 4 + np.argmax(diff[min_len // 5 * 4:])

    artifacts = [artifact0, artifact1, artifact2, artifact3, artifact4]

    for i in range(len(artifacts)):
        fig, axs = plt.subplots(2)

        axs[0].plot(ANNE_filt[artifacts[i] - sample_rate * wr:artifacts[i] + sample_rate * wr + 1])
        axs[0].title.set_text("ANNE")
        axs[1].plot(
            PSG_filt[artifacts[i] - sample_rate * wr - sample_lag:artifacts[i] + sample_rate * wr - sample_lag + 1])
        axs[1].plot(PSG_filt[artifacts[i] - sample_rate * wr:artifacts[i] + sample_rate * wr + 1], alpha=0.75)
        axs[1].title.set_text("PSG")
        plt.savefig(f"{DATA_DIR}/alignment_qc/{patientid}_{i}.png")
        plt.close()

    anne_len = len(ANNE_filt)
    psg_len = len(PSG_filt)
    n = 3

    fig_, axs_ = plt.subplots(n)

    for i in range(n):
        corr_ = signal.correlate(ANNE_filt[anne_len//n*i:anne_len//n*(i+1)], PSG_filt[psg_len//n*i:psg_len//n*(i+1)])
        lags_ = signal.correlation_lags(psg_len//n, anne_len//n)
        corr_ /= np.max(corr_)

        center_ = np.where(lags_ == 0)[0][0]

        corr_bounded_ = corr_[center_ - sample_rate * radius:center_ + sample_rate * radius]
        lags_bounded_ = lags_[center_ - sample_rate * radius:center_ + sample_rate * radius]
        axs_[i].plot(lags_bounded_ / 25, corr_bounded_)
        axs_[i].plot(lags_bounded / 25, corr_bounded, alpha=0.5)
        axs_[i].title.set_text(f"Segment {i}")

    plt.savefig(f"{DATA_DIR}/alignment_qc/{patientid}_seg_corr.png")
    plt.close()

    plt.plot(lags_bounded / 25, corr_bounded)
    plt.savefig(f"{DATA_DIR}/alignment_qc/{patientid}_corr.png")

    plt.close()

    return sample_lag


if __name__ == "__main__":
    file_list = os.listdir(DATA_DIR)
    # file_list = [132]
    shift_dict = {}

    for patient_id in file_list:
        try:
            patient_id = int(patient_id)
            print(patient_id)
            index, time, ecg, ppg, x_acc, y_acc, z_acc, enmo, z_angle, temp \
                = np.loadtxt(f"{DATA_DIR}/{patient_id}/ANNE.csv", delimiter=",", dtype="float",
                             skiprows=1, unpack=True,
                             usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9))
            PSG_signals, signal_headers, header = highlevel.read_edf(
                f"{DATA_DIR}/{patient_id}/PSG.edf")
            ANNE_ECG = ecg
            PSG_ECG = PSG_signals[7]

            # downsample PSG ecg from 256 hz to 25 hz
            PSG_ECG_resample = signal.resample_poly(PSG_ECG, 25, 256)

            shift_time_in_sec = align(patient_id, ANNE_ECG, PSG_ECG_resample) / 25
            shift_dict[patient_id] = shift_time_in_sec
            print(f"{shift_time_in_sec}\n")
        except Exception as e:
            print("something went wrong")
            print(str(e))

    print(shift_dict)
    json.dump(shift_dict, open("shift3.json", 'w'))
