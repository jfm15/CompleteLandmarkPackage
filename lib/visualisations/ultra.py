# These are here because they can be called from other files
from lib.visualisations.generic import gt_and_preds


def ddh_angles(ax, predicted_points, target_points):

    base_line_vector = predicted_points[1] - predicted_points[0]
    base_line_p1 = predicted_points[0] - base_line_vector
    base_line_p2 = predicted_points[0] + 4 * base_line_vector

    cartilage_roof_line_vector = predicted_points[4] - predicted_points[2]
    cartilage_roof_line_p1 = predicted_points[2] - cartilage_roof_line_vector
    cartilage_roof_line_p2 = predicted_points[2] + 2 * cartilage_roof_line_vector

    bony_roof_line_vector = predicted_points[3] - predicted_points[2]
    bony_roof_line_vector_p1 = predicted_points[2] - 2 * bony_roof_line_vector
    bony_roof_line_vector_p2 = predicted_points[2] + 2 * bony_roof_line_vector

    ax.plot([base_line_p1[0], base_line_p2[0]], [base_line_p1[1], base_line_p2[1]], color='red')
    ax.plot([cartilage_roof_line_p1[0], cartilage_roof_line_p2[0]],
             [cartilage_roof_line_p1[1], cartilage_roof_line_p2[1]], color='yellow')
    ax.plot([bony_roof_line_vector_p1[0], bony_roof_line_vector_p2[0]],
             [bony_roof_line_vector_p1[1], bony_roof_line_vector_p2[1]], color='cyan')

    ax.scatter(predicted_points[:, 0], predicted_points[:, 1], color='green', s=20)