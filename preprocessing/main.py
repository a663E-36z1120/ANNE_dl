# Code for integrating raw data into appropriate formats for later use
# @author: Andrew H. Zhang

import numpy as np
from pyedflib import highlevel
from matplotlib import pyplot as plt

DATA_DIR = "/mnt/Common/Data"
SAMP_RATE = 25  # Hz


def integrate_data(subject_id):
    # Read raw signals from csv
    index, time, ecg, ppg, x_acc, y_acc, z_acc, enmo, z_angle, \
        = np.loadtxt(f"{DATA_DIR}/{subject_id}/ANNE.csv", delimiter=",", dtype="float",
                     skiprows=1, unpack=True,
                     usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8))

    # Read additional features from edf
    signals_psg, signal_headers_psg, header_psg = highlevel.read_edf(f"{DATA_DIR}/{subject_id}/PSG.edf")
    ecg_psg = signals_psg[7]

    ns = 10000

    plt.plot(np.linspace(0,ns,25*ns), ecg[:ns*25], alpha=0.5)
    plt.plot(np.linspace(0, ns, 256 * ns), ecg_psg[:ns * 256], alpha=0.5)
    plt.show()

    pass


if __name__ == "__main__":
    integrate_data(132)
