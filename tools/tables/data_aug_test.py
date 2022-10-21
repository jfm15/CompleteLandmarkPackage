import numpy as np

no_aug_ceph_results = np.array([1.486, 1.494, 1.454, 1.630, 1.556, 1.578])
rot_ceph_results = np.array([1.486, 1.494, 1.454, 1.630, 1.556, 1.578])
rot_int_ceph_results = np.array([1.482, 1.439, 1.454])
rot_int_sf_ceph_results = np.array([1.421, 1.410, 1.420])
rot_int_sf_t_ceph_results = np.array([1.387, 1.368, 1.475])
#best_ceph_results = np.array([1.464, 1.415, 1.374])

no_aug_hand_results = np.array([0.667, 0.679, 0.685])
rot_hand_results = np.array([0, 0, 0])
rot_int_hand_results = np.array([0, 0, 0])
rot_int_sf_hand_results = np.array([0.645, 0.643, 0.650])
#rot_int_sf_t_hand_results = np.array([0.646, 0.645,  0.638])

no_aug_pelvis_results = np.array([2.343, 2.253, 2.328])
rot_pelvis_results = np.array([2.282, 2.193, 2.187])
rot_int_pelvis_results = np.array([1.976, 2.187, 2.276])
rot_int_sf_pelvis_results = np.array([2.177, 2.166, 2.241])
rot_int_sf_t_pelvis_results = np.array([2.172, 2.208, 2.211])
#best_pelvis_results = np.array([2.167, 2.246, 2.357])

no_aug_ultra_results = np.array([6.593, 7.114, 6.848])
rot_ultra_results = np.array([7.025, 6.787, 7.086])
rot_int_ultra_results = np.array([6.837, 6.912, 6.771])
rot_int_sf_ultra_results = np.array([6.485, 6.906, 7.162])
rot_int_sf_t_ultra_results = np.array([6.916, 7.003, 6.689])
#best_ultra_results = np.array([6.957, 6.959, 6.510])


def print_stats(arr):
    mean = np.mean(arr)
    std = np.std(arr)
    msg = "{:.3f}$\pm${:.3f}".format(mean, std)
    print(msg)


print("start")
for arr in [no_aug_ceph_results,
            rot_ceph_results,
            rot_int_ceph_results,
            rot_int_sf_ceph_results,
            rot_int_sf_t_ceph_results]:
    print_stats(arr)

print("start")
for arr in [no_aug_hand_results,
            rot_hand_results,
            rot_int_hand_results,
            rot_int_sf_hand_results]:
    print_stats(arr)

print("start")
for arr in [no_aug_pelvis_results,
            rot_pelvis_results,
            rot_int_pelvis_results,
            rot_int_sf_pelvis_results,
            rot_int_sf_t_pelvis_results]:
    print_stats(arr)

print("start")
for arr in [no_aug_ultra_results,
            rot_ultra_results,
            rot_int_ultra_results,
            rot_int_sf_ultra_results,
            rot_int_sf_t_ultra_results]:
    print_stats(arr)
