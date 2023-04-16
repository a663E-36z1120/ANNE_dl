import matplotlib.pyplot as plt
import numpy as np
from pyedflib import highlevel
from matplotlib import pyplot as plt
from math import ceil
from scipy import signal
from alignment import *

DATA_DIR = "/Users/lichunlin/Desktop/ANNE_data"
SAMP_RATE = 25  # Hz
WINDOW_LEN = 30  # Seconds
current_index = 5500

def manual_align_one_sequence(subject_id):
    index, time, anne_ecg, ppg, x_acc, y_acc, z_acc, enmo, z_angle, temp \
        = np.loadtxt(f"{DATA_DIR}/{subject_id}/ANNE.csv", delimiter=",", dtype="float32",
                     skiprows=1, unpack=True,
                     usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9))

    signals, signal_headers, header = highlevel.read_edf(
        f"{DATA_DIR}/{subject_id}/PSG.edf")
    psg_samp_rate = int(signal_headers[7]["sample_rate"])
    PSG_ECG = signals[7]
    PSG_thor_resample = signal.resample_poly(PSG_ECG, 25, psg_samp_rate)

    # Assuming your lists are named `data1` and `data2` and contain your data
    data1 = anne_ecg[5000:]  # Replace [...] with your actual data
    data2 = PSG_thor_resample[5000:]  # Replace [...] with your actual data

    # Plot the first 100 points for both data sets
    fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axs[0].plot(data1[:500])
    axs[0].set_title('Data 1')
    axs[1].plot(data2[:500])
    axs[1].set_title('Data 2')

    # Define a global variable to keep track of the current index


    # Define a function to show the next 100 points for both data sets when space is pressed
    def on_key_press(event):
        global current_index
        if event.key == ' ' and current_index < len(data1):
            next_index = min(current_index + 500, len(data1))
            axs[0].clear()
            axs[1].clear()
            axs[0].plot(data1[current_index:next_index])
            axs[0].set_title('Data 1')
            axs[1].plot(data2[current_index:next_index])
            axs[1].set_title('Data 2')
            plt.draw()
            current_index = next_index

    # Connect the key press event to the function we just defined
    fig.canvas.mpl_connect('key_press_event', on_key_press)

    # Show the plots
    plt.show()

if __name__ == "__main__":
    manual_align_one_sequence(324)
