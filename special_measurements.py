import torch

from utils import get_angle


def ultrasound_alpha_angle(points):

    baseline = points[1, :] - points[0, :]
    bony_roof_line = points[3, :] - points[2, :]

    return get_angle(baseline, bony_roof_line)


def ultrasound_beta_angle(points):

    baseline = points[1, :] - points[0, :]
    cartilage_roof_lines = points[4, :] - points[2, :]

    return get_angle(baseline, cartilage_roof_lines)


def ap_left_alpha_angle(points):
    center_of_circle = get_center_of_circle(points[10], points[11], points[12])
    center_of_circle = center_of_circle.to(points.device)
    neck_point = points[2]
    cam_point = points[3]

    neck_axis = center_of_circle - neck_point
    cam_axis = cam_point - center_of_circle

    return get_angle(cam_axis, neck_axis)


def ap_right_alpha_angle(points):
    center_of_circle = get_center_of_circle(points[25], points[26], points[27])
    center_of_circle = center_of_circle.to(points.device)
    neck_point = points[17]
    cam_point = points[18]

    neck_axis = center_of_circle - neck_point
    cam_axis = cam_point - center_of_circle

    return get_angle(cam_axis, neck_axis)


def ap_left_lce_angle(points):

    center_of_circle = get_center_of_circle(points[10], points[11], points[12])
    center_of_circle = center_of_circle.to(points.device)
    lat_point = points[0]

    lat_axis = lat_point - center_of_circle
    up_axis = torch.Tensor([0, -1]).to(points.device)

    return get_angle(lat_axis.float(), up_axis)


def ap_right_lce_angle(points):

    center_of_circle = get_center_of_circle(points[25], points[26], points[27])
    center_of_circle = center_of_circle.to(points.device)
    lat_point = points[15]

    lat_axis = lat_point - center_of_circle
    up_axis = torch.Tensor([0, -1]).to(points.device)

    return get_angle(lat_axis.float(), up_axis)


def get_center_of_circle(point1, point2, point3):
    x_1, y_1 = point1
    x_2, y_2 = point2
    x_3, y_3 = point3
    A = x_1 * (y_2 - y_3) - y_1 * (x_2 - x_3) + x_2 * y_3 - x_3 * y_2
    B = (x_1 * x_1 + y_1 * y_1) * (y_3 - y_2) + \
        (x_2 * x_2 + y_2 * y_2) * (y_1 - y_3) + (x_3 * x_3 + y_3 * y_3) * (y_2 - y_1)
    C = (x_1 * x_1 + y_1 * y_1) * (x_2 - x_3) + \
        (x_2 * x_2 + y_2 * y_2) * (x_3 - x_1) + (x_3 * x_3 + y_3 * y_3) * (x_1 - x_2)
    return torch.Tensor([-B / (2 * A), -C / (2 * A)])
