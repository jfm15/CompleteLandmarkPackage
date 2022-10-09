import numpy as np

from lib.utils import get_center_of_circle

# These are here because they can be called from other files


def clear_gt(ax, predicted_points, target_points):

    text_offsets = np.ones((len(target_points), 2)) * np.array([5, -5])
    text_offsets[4] = np.array([5, 15])
    text_offsets[10] = np.array([-35, -5])
    text_offsets[7] = np.array([-5, 25])
    text_offsets[8] = np.array([10, 15])

    ax.scatter(target_points[:, 0], target_points[:, 1], color='green', s=20)

    for i, positions in enumerate(target_points):
        text_position = positions + text_offsets[i]
        ax.text(text_position[0], text_position[1], "{}".format(i + 1), color="yellow", fontsize="large")