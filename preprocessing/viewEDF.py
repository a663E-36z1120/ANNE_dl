import numpy as np
import matplotlib.pyplot as plt
import mne

data = mne.io.read_raw_edf("/mnt/Common/Downloads/19-12-19-20_11_12.C823.L775.3-annotated.edf")
raw_data = data.get_data()
print(raw_data[27])
plt.plot(raw_data[27])
plt.show()

