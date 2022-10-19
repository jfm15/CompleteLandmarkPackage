import numpy as np

not_pretrained_ceph_results = np.array([1.587, 1.569, 1.599])
not_pretrained_hand_results = np.array([0.705, 0.710, 0.729])
not_pretrained_pelvis_results = np.array([2.789, 2.865, 2.748])
not_pretrained_ultra_results = np.array([6.954, 6.892, 6.895])


def print_stats(arr):
    mean = np.mean(arr)
    std = np.std(arr)
    msg = "{:.3f}$\pm${:.3f}".format(mean, std)
    print(msg)


for arr in [not_pretrained_ceph_results,
            not_pretrained_hand_results,
            not_pretrained_pelvis_results,
            not_pretrained_ultra_results]:
    print_stats(arr)
