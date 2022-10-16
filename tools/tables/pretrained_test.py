import numpy as np

pretrained_ceph_results = np.array([1.39, 1.40, 1.40])
not_pretrained_ceph_results = np.array([1.54, 1.62, 1.56])

pretrained_hand_results = np.array([0.65])
not_pretrained_hand_results = np.array([0.73])

pretrained_pelvis_results = np.array([2.28, 2.33, 2.15])
not_pretrained_pelvis_results = np.array([2.76, 2.81, 2.82])

pretrained_ultra_results = np.array([6.56, 6.86, 6.71])
not_pretrained_ultra_results = np.array([6.97, 7.21, 7.05])


def print_stats(arr):
    mean = np.mean(arr)
    std = np.std(arr)
    print(mean, std)


for arr in [pretrained_ceph_results,
            not_pretrained_ceph_results,
            pretrained_hand_results,
            not_pretrained_hand_results,
            pretrained_pelvis_results,
            not_pretrained_pelvis_results,
            pretrained_ultra_results,
            not_pretrained_ultra_results]:
    print_stats(arr)
