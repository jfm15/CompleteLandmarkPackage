import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

csv_paths = ['csvs/ere_correlation_vs_sigma_ceph.csv',
             'csvs/ere_correlation_vs_sigma_hand.csv',
             'csvs/ere_correlation_vs_sigma_ap.csv',
             'csvs/ere_correlation_vs_sigma_ultra.csv'
]

names = ['Cephalometric', 'Hand', 'X-ray Pelvis', 'Ultrasound Hip']

for csv_path, name in zip(csv_paths, names):
    df = pd.read_csv(csv_path)
    df = df.sort_values(by=['DATASET.SIGMA'])
    sigma_values = df['DATASET.SIGMA'].to_numpy()
    np_values = df['radial_ere_correlation'].to_numpy()
    plt.plot(sigma_values, np_values, label=name)

plt.grid(True)
plt.legend(fontsize=20, loc="upper right")
plt.xlabel("\u03C3 of training heatmaps", fontsize=25)
plt.ylabel("ERE To Radial Error Correlation", fontsize=25)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.show()
