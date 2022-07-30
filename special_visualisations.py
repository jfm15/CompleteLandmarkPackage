import os
import numpy as np
import matplotlib.pyplot as plt


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