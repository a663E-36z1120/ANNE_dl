import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
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


def align(patientid, ANNE, PSG):
    # fs = sample_rate  # sample rate, Hz
    # cutoff = 4  # desired cutoff frequency of the filter, Hz
    # order = 2  # sin wave can be approx represented as quadratic
    #
    # ANNE_filt = butter_lowpass_filter(ANNE, cutoff, fs, order)
    # PSG_filt = butter_lowpass_filter(PSG, cutoff, fs, order)

    ANNE_filt = ANNE
    PSG_filt = PSG

    corr = signal.correlate(ANNE_filt, PSG_filt)
    lags = signal.correlation_lags(len(PSG_filt), len(ANNE_filt))
    corr /= np.max(corr)
    len_corr = len(corr)

    center = np.where(lags == 0)[0][0]
    radius = 120

    corr_bounded = corr[center - sample_rate * radius:center + sample_rate * radius]
    lags_bounded = lags[center - sample_rate * radius:center + sample_rate * radius]

    sample_lag = lags_bounded[np.argmax(corr_bounded)]

    # Generate some plots around artifacts to assess quality of alignment

    wr = 15  # window radius around each artifact to generate the plot for

    artifacts = [np.argmax(PSG_filt), np.argmax(ANNE_filt), len(ANNE_filt) // 2]

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
        except:
            print("something went wrong")

    print(shift_dict)
    json.dump(shift_dict, open("shift2.json", 'w'))
#