import numpy as np

bce_ceph_results = np.array([1.430, 1.375, 1.382])
bce_hand_results = np.array([0.647, 0.646, 0.650])
bce_pelvis_results = np.array([2.431, 2.154, 2.380])
bce_ultra_results = np.array([6.886, 6.613, 6.977])


def print_stats(arr):
    mean = np.mean(arr)
    std = np.std(arr)
    msg = "{:.3f}$\pm${:.3f}".format(mean, std)
    print(msg)


print("start")
for arr in [bce_ceph_results,
            bce_hand_results,
            bce_pelvis_results,
            bce_ultra_results]:
    print_stats(arr)
