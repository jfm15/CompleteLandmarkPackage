import numpy as np
import matplotlib.pyplot as plt



ceph_results = np.array([
    [27.05,  29.32, 30.32],
    [28.68, 28.21, 29.47],
    [27.95, 27.58, 28.00],
    [25.42, 24.58, 24.47],
    [22.84, 19.21, 21.53],
    [19.26, 18.26, 21.63],
    [19.58, 18.63, 18.84],
    [15.37, 17.26, 19.16]
])

ceph_results_high = np.array([
    [99.84, 99.79, 99.79],
    [99.84, 99.89, 99.84],
    [99.89, 99.89, 99.95],
    [99.84, 99.89, 99.89],
    [99.95, 99.89, 99.95],
    [99.89, 99.89, 99.95],
    [99.89, 99.95, 99.89],
    [99.95, 99.95, 100.00]
])

hand_results = np.array([
])

pelvis_results = np.array([
])

ultra_results = np.array([
])

def print_state(results):
    print("start")
    means = np.mean(results, axis=1)
    for mean in means:
        msg = "{:.3f} {:.3f}\%".format(round(mean / 0.05263157894), mean)
        print(msg)

for arr in [ceph_results, ceph_results_high]:
    print_state(arr)