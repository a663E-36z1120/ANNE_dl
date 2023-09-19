import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, lfilter
from pyedflib import highlevel
from scipy import signal
import os
import json
from datetime import datetime, timedelta

DATA_DIR = "/mnt/Common/data-full"
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
    fs = sample_rate  # sample rate, Hz
    order = 2  # sin wave can be approx represented as quadratic

    # ANNE_filt = butter_bandpass_filter(ANNE, 0.6, 2, fs, order)
    # PSG_filt = butter_lowpass_filter(PSG, 2, fs, order)

    ANNE_filt = ANNE
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
        fig, axs = plt.subplots(2, sharex=True)

        t_axis = np.arange(artifacts[i] - sample_rate * wr, artifacts[i] + sample_rate * wr + 1) / 25

        axs[0].plot(t_axis, ANNE_filt[artifacts[i] - sample_rate * wr:artifacts[i] + sample_rate * wr + 1])
        axs[0].title.set_text("ANNE One")
        axs[1].plot(t_axis,
            PSG_filt[artifacts[i] - sample_rate * wr - sample_lag:artifacts[i] + sample_rate * wr - sample_lag + 1], label="After Alignment")
        axs[1].plot(t_axis, PSG_filt[artifacts[i] - sample_rate * wr:artifacts[i] + sample_rate * wr + 1], alpha=0.75, label="Before Alignment")
        axs[1].title.set_text("PSG")
        axs[1].legend()
        axs[1].set_ylabel('ECG (mV)')
        axs[0].set_ylabel('ECG (mV)')
        axs[1].set_xlabel('time (s)')
        plt.tight_layout()
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

    plt.figure(figsize=(5, 3))
    plt.plot(lags_bounded / 25, corr_bounded)
    plt.xlabel("Signal Misalignment (s)")
    plt.ylabel("Cross Correlation Magnitude")
    plt.tight_layout()
    plt.savefig(f"{DATA_DIR}/alignment_qc/{patientid}_corr.png")

    plt.close()

    return sample_lag

def find_anne_file(directory_path):
    anne_file = None
    for filename in os.listdir(directory_path):
        if filename.endswith(".edf") and "PSG" not in filename and "vital" not in filename:
            anne_file = os.path.join(directory_path, filename)
            break
    return str(anne_file)

def find_psg_file(directory_path):
    psg_file = None
    for filename in os.listdir(directory_path):
        if filename.endswith(".edf") and "PSG" in filename:
            psg_file = os.path.join(directory_path, filename)
            break
    return str(psg_file)


def extract_anne_time(input_string):
    date_time_str = input_string.split('.')[0]  # Extract the part before the first dot
    date_time_obj = datetime.strptime(date_time_str, "%y-%m-%d-%H_%M_%S")
    # formatted_date = date_time_obj.strftime("%Y, %B, %d %H:%M:%S")
    return date_time_obj - timedelta(hours=5)

def clip_array_within_std(arr):
    """
    Clips a numpy array between plus or minus 1 standard deviation from its mean.

    Parameters:
        arr (numpy.ndarray): Input numpy array.

    Returns:
        numpy.ndarray: Clipped numpy array.
    """
    mean = np.mean(arr)
    std = np.std(arr)
    lower_bound = mean - std
    upper_bound = mean + std
    clipped_arr = np.clip(arr, lower_bound, upper_bound)
    return clipped_arr - mean


if __name__ == "__main__":


    # file_list = os.listdir(DATA_DIR)
    file_list = ["Subject139"]
    shift_dict = {}

    for patient_id in file_list:
        # try:
            anne_file = find_anne_file(f"{DATA_DIR}/{patient_id}")
            psg_file = find_psg_file(f"{DATA_DIR}/{patient_id}")

            # psg_file = f"{DATA_DIR}/{patient_id}/191217_2648723_ANNE_PSGfiltered.edf"
            # patient_id = int(patient_id)
            # print(patient_id)
            # index, time, ecg, ppg, x_acc, y_acc, z_acc, enmo, z_angle, temp \
            #     = np.loadtxt(f"{DATA_DIR}/{patient_id}/ANNE.csv", delimiter=",", dtype="float",
            #                  skiprows=1, unpack=True,
            #                  usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9))
            ANNE_signals, signal_headers_, header_anne = highlevel.read_edf(
                f"{anne_file}")
            PSG_signals, signal_headers, header_psg = highlevel.read_edf(
                f"{psg_file}")

            anne_date = extract_anne_time(os.path.basename(anne_file))
            psg_date = header_psg["startdate"]

            time_diff = int((psg_date - anne_date).total_seconds())
            #
            ANNE_ECG = ANNE_signals[0][time_diff*128:]
            PSG_ECG = PSG_signals[14]


            # downsample PSG ecg from 256 hz to 25 hz
            PSG_ECG_resample = signal.resample_poly(PSG_ECG, 25, 256)
            ANNE_ECG_resample = signal.resample_poly(ANNE_ECG, 25, 128)


            fig, axs = plt.subplots(2, sharex="all")
            axs[0].plot(ANNE_ECG_resample)
            axs[1].plot(PSG_ECG_resample)
            plt.show()

            shift_time_in_sec = align(patient_id, ANNE_ECG_resample, PSG_ECG_resample) / 25
            shift_dict[patient_id] = shift_time_in_sec
            print(f"{shift_time_in_sec}\n")
        # except Exception as e:
        #     print(f"something went wrong for {patient_id}")
        #     print(str(e))


    print(shift_dict)
    json.dump(shift_dict, open("shift3.json", 'w'))
