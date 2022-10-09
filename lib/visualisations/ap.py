import numpy as np

from lib.utils import get_center_of_circle

# These are here because they can be called from other files


def alpha_angles(ax, predicted_points, target_points):

    important_indices = [2, 3, 10, 11, 12, 17, 18, 25, 26, 27]
    predicted_points = np.take(predicted_points, important_indices, axis=0)
    target_points = np.take(target_points, important_indices, axis=0)

    #ax.scatter(predicted_points[:, 0], predicted_points[:, 1], color='red', s=5)
    #ax.scatter(target_points[:, 0], target_points[:, 1], color='green', s=5)


    # Find center of the circle
    centers = []
    for i in [2, 7]:
        centers.append(get_center_of_circle(predicted_points[i],
                                            predicted_points[i + 1],
                                            predicted_points[i + 2]).numpy())
    centers = np.array(centers)

    # left
    ax.plot([centers[0, 0], predicted_points[0, 0]], [centers[0, 1], predicted_points[0, 1]], color='red')
    ax.plot([centers[0, 0], predicted_points[1, 0]], [centers[0, 1], predicted_points[1, 1]], color='red')

    # right
    ax.plot([centers[1, 0], predicted_points[5, 0]], [centers[1, 1], predicted_points[5, 1]], color='red')
    ax.plot([centers[1, 0], predicted_points[6, 0]], [centers[1, 1], predicted_points[6, 1]], color='red')


    #for i, positions in enumerate(target_points):
    #    idx = important_indices[i]
    #    ax.text(positions[0], positions[1], "{}".format(idx + 1), color="yellow", fontsize="small")


def lce_angles(ax, predicted_points, target_points):

    centers = []
    for i in [10, 25]:
        centers.append(get_center_of_circle(predicted_points[i],
                                            predicted_points[i + 1],
                                            predicted_points[i + 2]).numpy())
    centers = np.array(centers)

    # left
    ax.plot([centers[0, 0], predicted_points[0, 0]], [centers[0, 1], predicted_points[0, 1]], color='red')
    ax.plot([centers[0, 0], centers[0, 0]], [centers[0, 1], centers[0, 1] - 50], color='red')

    # right
    ax.plot([centers[1, 0], predicted_points[15, 0]], [centers[1, 1], predicted_points[15, 1]], color='red')
    ax.plot([centers[1, 0], centers[1, 0]], [centers[1, 1], centers[1, 1] - 50], color='red')


def neck_shaft_angles(ax, predicted_points, target_points):

    # left
    ax.plot([predicted_points[2, 0], predicted_points[5, 0]],
            [predicted_points[2, 1], predicted_points[5, 1]], color='red')
    ax.plot([predicted_points[5, 0], predicted_points[6, 0]],
            [predicted_points[5, 1], predicted_points[6, 1]], color='red')

    # right
    ax.plot([predicted_points[17, 0], predicted_points[20, 0]],
            [predicted_points[17, 1], predicted_points[20, 1]], color='red')
    ax.plot([predicted_points[20, 0], predicted_points[21, 0]],
            [predicted_points[20, 1], predicted_points[21, 1]], color='red')


def pelvic_tilt(ax, predicted_points, target_points):

    ax.plot([predicted_points[9, 0], predicted_points[24, 0]],
             [predicted_points[9, 1], predicted_points[24, 1]], color='red')


def acetabular_indices(ax, predicted_points, target_points):

    # left
    ax.plot([predicted_points[0, 0], predicted_points[1, 0]],
            [predicted_points[0, 1], predicted_points[1, 1]], color='red')
    ax.plot([predicted_points[1, 0], predicted_points[1, 0] + 50],
            [predicted_points[1, 1], predicted_points[1, 1]], color='red')

    # right
    ax.plot([predicted_points[15, 0], predicted_points[16, 0]],
            [predicted_points[15, 1], predicted_points[16, 1]], color='red')
    ax.plot([predicted_points[16, 0], predicted_points[16, 0] - 50],
            [predicted_points[16, 1], predicted_points[16, 1]], color='red')