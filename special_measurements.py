from utils import get_angle


def ultrasound_alpha_angle(points):

    baseline = points[1, :] - points[0, :]
    bony_roof_line = points[3, :] - points[2, :]

    return get_angle(baseline, bony_roof_line)


def ultrasound_beta_angle(points):

    baseline = points[1, :] - points[0, :]
    cartilage_roof_lines = points[4, :] - points[2, :]

    return get_angle(baseline, cartilage_roof_lines)
