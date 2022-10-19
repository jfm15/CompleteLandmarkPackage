import numpy as np

no_aug_ceph_results = np.array([1.469, 1.430, 1.482])
rot_ceph_results = np.array([1.562, 1.508, 1.535])
rot_int_ceph_results = np.array([1.404, 1.411, 1.374])
rot_int_sf_ceph_results = np.array([1.386, 1.375, 1.392])
rot_int_sf_t_ceph_results = np.array([1.415, 1.407, 1.409])

no_aug_hand_results = np.array([0.666, 0.693, 0.685])
rot_hand_results = np.array([0, 0, 0])
rot_int_hand_results = np.array([0, 0, 0])
rot_int_sf_hand_results = np.array([0.645, 0.643, 0.650])
rot_int_sf_t_hand_results = np.array([0.646, 0.645,  0.638])

no_aug_pelvis_results = np.array([2.171, 2.545, 3.160])
rot_pelvis_results = np.array([2.311, 2.146, 2.173])
rot_int_pelvis_results = np.array([2.121, 2.161, 2.204])
rot_int_sf_pelvis_results = np.array([2.063, 2.165, 2.085])
rot_int_sf_t_pelvis_results = np.array([2.320, 2.331, 2.064])

no_aug_ultra_results = np.array([6.866, 6.948, 6.927])
rot_ultra_results = np.array([6.921, 6.661, 7.067])
rot_int_ultra_results = np.array([6.826, 6.703, 6.812])
rot_int_sf_ultra_results = np.array([6.970, 6.677, 8.089])
rot_int_sf_t_ultra_results = np.array([6.590, 6.513, 6.631])


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
            rot_int_sf_hand_results,
            rot_int_sf_t_hand_results]:
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
