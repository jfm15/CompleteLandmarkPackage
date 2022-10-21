import numpy as np

bce_ceph_results = np.array([1.486, 1.494, 1.454])
bce_hand_results = np.array([0.667, 0.679, 0.685])
bce_pelvis_results = np.array([2.343, 2.253, 2.328])
bce_ultra_results = np.array([6.593, 7.114, 6.848])


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
