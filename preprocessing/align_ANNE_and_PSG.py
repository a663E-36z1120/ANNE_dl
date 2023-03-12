import numpy as np
from signal_alignment import phase_align
from scipy.ndimage import shift
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
    ANNE_signal = ANNE.tolist()
    PSG_signal = PSG.tolist()

    # ANNE_align_segment = ANNE_signal[start:end]
    # PSG_signal_segment = PSG_signal[start:end]

    # ANNE_align_segment = ANNE_signal[0:300000]
    # PSG_signal_segment = PSG_signal[0:300000]

    ANNE_align_segment = ANNE_signal
    PSG_signal_segment = PSG_signal

    fs = 25         # sample rate, Hz
    cutoff = 1     # desired cutoff frequency of the filter, Hz
    order = 2       # sin wave can be approx represented as quadratic

    ANNE_after_low_pass_filter = butter_lowpass_filter(ANNE_align_segment, cutoff, fs, order)
    PSG_after_low_pass_filter = butter_lowpass_filter(PSG_signal_segment, cutoff, fs, order)

    phase_shift_for_low_passed_data = phase_align(ANNE_after_low_pass_filter, PSG_after_low_pass_filter, (0, 10000))
    print("patient", patientid, "phase shift value to align is", phase_shift_for_low_passed_data / sample_rate)

    # _, axis = plt.subplots(1, 2)
    # axis[0].plot(ANNE_align_segment)
    # axis[0].set_title("ANNE_align_segment")
    # axis[1].plot(shift(ANNE_after_low_pass_filter, phase_shift_for_low_passed_data, mode='nearest'))
    # axis[1].set_title("PSG aligned")
    # plt.show()

    return phase_shift_for_low_passed_data


if __name__ == "__main__":
    file_list = os.listdir(DATA_DIR)
    # file_list = [223]
    # file_without_hidden = [f for f in file_list if not f.startswith('.')]
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
            print(shift_time_in_sec)
        except:
            print("Something went wrong")

    print(shift_dict)
    json.dump(shift_dict, open("shift.json", 'w'))
