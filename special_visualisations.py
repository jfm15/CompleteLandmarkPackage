import os
import numpy as np
import matplotlib.pyplot as plt
import torch

from special_measurements import get_center_of_circle


def ultrasound_ddh_angles(image, predicted_points, target_points, save=False, save_path=""):

        plt.imshow(image[0], cmap='gray')

        plt.scatter(predicted_points[:, 0], predicted_points[:, 1], color='red', s=5)
        plt.scatter(target_points[:, 0], target_points[:, 1], color='green', s=5)

        for i, positions in enumerate(target_points):
            plt.text(positions[0], positions[1], "{}".format(i + 1), color="yellow", fontsize="small")

        base_line_vector = predicted_points[1] - predicted_points[0]
        base_line_p1 = predicted_points[0] - base_line_vector
        base_line_p2 = predicted_points[0] + 4 * base_line_vector

        cartilage_roof_line_vector = predicted_points[4] - predicted_points[2]
        cartilage_roof_line_p1 = predicted_points[2] - cartilage_roof_line_vector
        cartilage_roof_line_p2 = predicted_points[2] + 2 * cartilage_roof_line_vector

        bony_roof_line_vector = predicted_points[3] - predicted_points[2]
        bony_roof_line_vector_p1 = predicted_points[2] - 2 * bony_roof_line_vector
        bony_roof_line_vector_p2 = predicted_points[2] + 2 * bony_roof_line_vector

        plt.plot([base_line_p1[0], base_line_p2[0]], [base_line_p1[1], base_line_p2[1]], color='red')
        plt.plot([cartilage_roof_line_p1[0], cartilage_roof_line_p2[0]],
                 [cartilage_roof_line_p1[1], cartilage_roof_line_p2[1]], color='yellow')
        plt.plot([bony_roof_line_vector_p1[0], bony_roof_line_vector_p2[0]],
                 [bony_roof_line_vector_p1[1], bony_roof_line_vector_p2[1]], color='cyan')

        plt.axis('off')

        if save:
                plt.savefig(save_path)
                plt.close()
        else:
                plt.show()


def ap_alpha_angles(image, predicted_points, target_points, save=False, save_path=""):

        plt.imshow(image[0], cmap='gray')

        important_indices = [2, 3, 10, 11, 12, 17, 18, 25, 26, 27]
        predicted_points = np.take(predicted_points, important_indices, axis=0)
        target_points = np.take(target_points, important_indices, axis=0)

        plt.scatter(predicted_points[:, 0], predicted_points[:, 1], color='red', s=5)
        plt.scatter(target_points[:, 0], target_points[:, 1], color='green', s=5)

        # Find center of the circle
        centers = []
        for i in [2, 7]:
             centers.append(get_center_of_circle(predicted_points[i],
                                                 predicted_points[i + 1],
                                                 predicted_points[i + 2]).numpy())
        centers = np.array(centers)

        # left
        plt.plot([centers[0, 0], predicted_points[0, 0]], [centers[0, 1], predicted_points[0, 1]], color='red')
        plt.plot([centers[0, 0], predicted_points[1, 0]], [centers[0, 1], predicted_points[1, 1]], color='red')

        # right
        plt.plot([centers[1, 0], predicted_points[5, 0]], [centers[1, 1], predicted_points[5, 1]], color='red')
        plt.plot([centers[1, 0], predicted_points[6, 0]], [centers[1, 1], predicted_points[6, 1]], color='red')

        for i, positions in enumerate(target_points):
            idx = important_indices[i]
            plt.text(positions[0], positions[1], "{}".format(idx + 1), color="yellow", fontsize="small")

        plt.axis('off')

        if save:
                plt.savefig(save_path)
                plt.close()
        else:
                plt.show()


def ap_lce_angles(image, predicted_points, target_points, save=False, save_path=""):

        plt.imshow(image[0], cmap='gray')

        centers = []
        for i in [10, 25]:
             centers.append(get_center_of_circle(predicted_points[i],
                                                 predicted_points[i + 1],
                                                 predicted_points[i + 2]).numpy())
        centers = np.array(centers)

        # left
        plt.plot([centers[0, 0], predicted_points[0, 0]], [centers[0, 1], predicted_points[0, 1]], color='red')
        plt.plot([centers[0, 0], centers[0, 0]], [centers[0, 1], centers[0, 1] - 50], color='red')

        # right
        plt.plot([centers[1, 0], predicted_points[15, 0]], [centers[1, 1], predicted_points[15, 1]], color='red')
        plt.plot([centers[1, 0], centers[1, 0]], [centers[1, 1], centers[1, 1] - 50], color='red')

        plt.axis('off')

        if save:
                plt.savefig(save_path)
                plt.close()
        else:
                plt.show()


def ap_neck_shaft_angles(image, predicted_points, target_points, save=False, save_path=""):

        plt.imshow(image[0], cmap='gray')

        # left
        plt.plot([predicted_points[2, 0], predicted_points[5, 0]],
                 [predicted_points[2, 1], predicted_points[5, 1]], color='red')
        plt.plot([predicted_points[5, 0], predicted_points[6, 0]],
                 [predicted_points[5, 1], predicted_points[6, 1]], color='red')

        # right
        plt.plot([predicted_points[17, 0], predicted_points[20, 0]],
                 [predicted_points[17, 1], predicted_points[20, 1]], color='red')
        plt.plot([predicted_points[20, 0], predicted_points[21, 0]],
                 [predicted_points[20, 1], predicted_points[21, 1]], color='red')

        plt.axis('off')

        if save:
                plt.savefig(save_path)
                plt.close()
        else:
                plt.show()
