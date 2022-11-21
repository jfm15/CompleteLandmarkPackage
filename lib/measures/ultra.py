from lib.utils import get_angle


def alpha_angle(points):

    baseline = points[1, :] - points[0, :]
    bony_roof_line = points[3, :] - points[2, :]

    return get_angle(baseline, bony_roof_line)


def beta_angle(points):

    baseline = points[1, :] - points[0, :]
    cartilage_roof_lines = points[4, :] - points[2, :]

    return get_angle(baseline, cartilage_roof_lines)


def ddh(points):

    aa = alpha_angle(points)
    ba = beta_angle(points)

    if aa < 60:
        return [1]
    else:
        return [0]
