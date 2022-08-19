import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial import procrustes
from scipy.linalg import orthogonal_procrustes


def scale_matrix(scale):
    return np.array([[scale, 0], [0, scale]])


def rotation_matrix(degrees):
    radians = degrees * np.pi / 180
    return np.array([[np.cos(radians), np.sin(radians)], [-np.sin(radians), np.cos(radians)]])


a = np.array([[1, 3], [1, 2], [1, 1], [2, 1]], 'd')
# b = np.array([[4, -2], [4, -4], [4, -6], [2, -6]], 'd')
scale = scale_matrix(3)
rotation = rotation_matrix(45)
x_reflection = np.array([[1, 0], [0, -1]])

normalized, _, _ = procrustes(a, a)

transformed = np.matmul(np.matmul(np.matmul(normalized, scale), rotation), x_reflection)

error = transformed + (np.random.rand(*transformed.shape) - 0.5) * 0.2

R, S = orthogonal_procrustes(normalized, error)

remapped = np.matmul(np.matmul(normalized, R), scale_matrix(S))

plt.scatter(normalized[:, 0], normalized[:, 1], label="base")
plt.scatter(transformed[:, 0], transformed[:, 1], label="transformed")
plt.scatter(error[:, 0], error[:, 1], label="error")
plt.scatter(remapped[:, 0], remapped[:, 1], label="remapped")
#plt.scatter(mapped[:, 0], mapped[:, 1], label="mapped")
#plt.scatter(mtx1[:, 0], mtx1[:, 1], label="mtx1")
#plt.scatter(mtx2[:, 0], mtx2[:, 1], label="mtx2")
plt.legend()
plt.show()