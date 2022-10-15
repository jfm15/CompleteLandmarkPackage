import numpy as np

u_net_plus_ceph_results = np.array([1.39, 1.40, 1.40])
u_net_ceph_results = np.array([1.41, 1.41, 1.42])

u_net_plus_hand_results = np.array([0.65])
u_net_hand_results = np.array([0.65, 0.65, 0.66])

u_net_plus_pelvis_results = np.array([2.28, 2.33, 2.15])
u_net_pelvis_results = np.array([2.30, 2.41, 2.30])

u_net_plus_ultra_results = np.array([6.56, 6.86, 6.71])
u_net_ultra_results = np.array([6.66, 7.01, 6.74])

# averages
u_net_plus_ceph_averaged = np.mean(u_net_plus_ceph_results)
u_net_plus_ceph_std = np.std(u_net_plus_ceph_results)

u_net_ceph_averaged = np.mean(u_net_ceph_results)
u_net_ceph_std = np.std(u_net_ceph_results)

u_net_plus_hand_averaged = np.mean(u_net_plus_hand_results)
u_net_plus_hand_std = np.std(u_net_plus_hand_results)

u_net_hand_averaged = np.mean(u_net_hand_results)
u_net_hand_std = np.std(u_net_hand_results)

u_net_plus_pelvis_averaged = np.mean(u_net_plus_pelvis_results)
u_net_plus_pelvis_std = np.std(u_net_plus_pelvis_results)

u_net_pelvis_averaged = np.mean(u_net_pelvis_results)
u_net_pelvis_std = np.std(u_net_pelvis_results)

u_net_plus_ultra_averaged = np.mean(u_net_plus_ultra_results)
u_net_plus_ultra_std = np.std(u_net_plus_ultra_results)

u_net_ultra_averaged = np.mean(u_net_ultra_results)
u_net_ultra_std = np.std(u_net_ultra_results)

print(u_net_plus_ceph_averaged, u_net_plus_ceph_std)
print(u_net_ceph_averaged, u_net_ceph_std)

print(u_net_plus_hand_averaged, u_net_plus_hand_std)
print(u_net_hand_averaged, u_net_hand_std)

print(u_net_plus_pelvis_averaged, u_net_plus_pelvis_std)
print(u_net_pelvis_averaged, u_net_pelvis_std)

print(u_net_plus_ultra_averaged, u_net_plus_ultra_std)
print(u_net_ultra_averaged, u_net_ultra_std)
