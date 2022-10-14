import numpy as np
import matplotlib.pyplot as plt

ceph_results = np.array([
    [1.42, 1.54, 1.48],
    [1.47, 1.37, 1.39],
    [1.43, 1.41, 1.47],
    [1.39, 1.40, 1.40],
    [1.43, 1.40, 1.43],
    [1.42, 1.44, 1.43],
    [1.47, 1.43, 1.48],
    [1.47, 1.51, 1.52]
])

hand_results = np.array([
    [0.66],
    [0.65],
    [0.65],
    [0.66],
    [0.69],
    [0.69],
    [0.71],
    [0.71]
])

pelvis_results = np.array([
    [2.31, 2.27, 2.34],
    [2.26, 2.29, 2.23],
    [2.28, 2.29, 2.22],
    [2.28, 2.33, 2.15],
    [2.22, 2.37, 2.26],
    [2.30, 2.30, 2.39],
    [2.37, 2.35, 2.42],
    [2.44, 2.47, 2.50]
])

ultra_results = np.array([
    [7.41, 7.05, 7.21],
    [7.15, 6.95, 7.39],
    [6.76, 7.13, 7.00],
    [6.69, 7.07, 6.92],
    [7.15, 7.05, 6.80],
    [7.39, 6.60, 6.87],
    [6.56, 6.86, 6.71],
    [7.08, 7.16, 6.79]
])

ceph_averaged = np.mean(ceph_results, axis=1)
hand_averaged = np.mean(hand_results, axis=1)
pelvis_averaged = np.mean(pelvis_results, axis=1)
ultra_averaged = np.mean(ultra_results, axis=1)

print(ceph_averaged)
print(hand_averaged)
print(pelvis_averaged)
print(ultra_averaged)

plt.plot(range(len(ceph_averaged)), ceph_averaged)
plt.plot(range(len(hand_averaged)), hand_averaged)
plt.plot(range(len(pelvis_averaged)), pelvis_averaged)
plt.plot(range(len(ultra_averaged)), ultra_averaged)
plt.show()