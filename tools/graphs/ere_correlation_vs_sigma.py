import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('ece_vs_epoch.csv')

# column_names = list(df.columns)
# relevant_column_names = [x for x in column_names if x[-3:] == 'ece']

relevant_column_names = ['ultra_hip_unetplus_sigma_0 - ece', 'hands_unetplus_sigma_0 - ece',
                         'ap_pelvis_unetplus_sigma_0 - ece', 'ceph_unetplus_sigma_0 - ece']


labels = ["Ultrasound Hip (σ = 0)", "Hand (σ = 0)",
          "X-Ray Pelvis (σ = 0)", "Cephalometric (σ = 0)"]
np_values = df[relevant_column_names].to_numpy().T

for label, ys in zip(labels, np_values):
    xs = range(len(ys))
    plt.plot(xs, ys, label=label)

plt.grid(True)
plt.legend(fontsize=20, loc="upper right")
plt.xlabel("Epochs", fontsize=25)
plt.ylabel("Expected Calibration Error (ECE)", fontsize=25)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.show()