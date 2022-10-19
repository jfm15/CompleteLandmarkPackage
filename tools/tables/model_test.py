import numpy as np

u_net_ceph_results = np.array([1.404, 1.408, 1.427])
u_net_hand_results = np.array([0.656, 0.657, 0.660])
u_net_pelvis_results = np.array([2.275, 2.232, 2.323])
u_net_ultra_results = np.array([7.118, 7.413, 6.803])

# averages
def print_stats(arr):
    mean = np.mean(arr)
    std = np.std(arr)
    msg = "{:.3f}$\pm${:.3f}".format(mean, std)
    print(msg)


for arr in [u_net_ceph_results,
            u_net_hand_results,
            u_net_pelvis_results,
            u_net_ultra_results]:
    print_stats(arr)
