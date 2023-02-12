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
    mre_values = df['MRE'].to_numpy()
    correlation_values = df['radial_ere_correlation'].to_numpy()
    plt.scatter(mre_values, correlation_values)
    plt.show()

'''
plt.grid(True)
plt.legend(fontsize=20, loc="upper right")
plt.xlabel("Mean Radial Error", fontsize=25)
plt.ylabel("Radial Error To ERE Correlation", fontsize=25)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.show()
'''
