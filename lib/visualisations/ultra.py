# These are here because they can be called from other files
from lib.visualisations.generic import gt_and_preds


def alpha_angle(ax, points, color):

    base_line_vector = points[1] - points[0]
    base_line_p1 = points[0] - base_line_vector
    base_line_p2 = points[0] + 4 * base_line_vector

    bony_roof_line_vector = points[3] - points[2]
    bony_roof_line_vector_p1 = points[2] - 2 * bony_roof_line_vector
    bony_roof_line_vector_p2 = points[2] + 2 * bony_roof_line_vector

    ax.plot([base_line_p1[0], base_line_p2[0]], [base_line_p1[1], base_line_p2[1]], color=color)
    ax.plot([bony_roof_line_vector_p1[0], bony_roof_line_vector_p2[0]],
            [bony_roof_line_vector_p1[1], bony_roof_line_vector_p2[1]], color=color)


def beta_angle(ax, points, color):

    base_line_vector = points[1] - points[0]
    base_line_p1 = points[0] - base_line_vector
    base_line_p2 = points[0] + 4 * base_line_vector

    cartilage_roof_line_vector = points[4] - points[2]
    cartilage_roof_line_p1 = points[2] - cartilage_roof_line_vector
    cartilage_roof_line_p2 = points[2] + 2 * cartilage_roof_line_vector

    ax.plot([base_line_p1[0], base_line_p2[0]], [base_line_p1[1], base_line_p2[1]], color=color)
    ax.plot([cartilage_roof_line_p1[0], cartilage_roof_line_p2[0]],
             [cartilage_roof_line_p1[1], cartilage_roof_line_p2[1]], color=color)


def compare_alpha_angle(ax, predicted_points, target_points):

    alpha_angle(ax, target_points, "lime")
    alpha_angle(ax, predicted_points, "red")


def compare_beta_angle(ax, predicted_points, target_points):

    beta_angle(ax, target_points, "lime")
    beta_angle(ax, predicted_points, "red")
