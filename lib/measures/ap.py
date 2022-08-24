import torch

from lib.utils import get_angle
from lib.utils import get_center_of_circle


def alpha_angle(points, startIdx):
    center_of_circle = get_center_of_circle(points[startIdx + 10], points[startIdx + 11], points[startIdx + 12])
    center_of_circle = center_of_circle.to(points.device)
    neck_point = points[startIdx + 2]
    cam_point = points[startIdx + 3]

    neck_axis = center_of_circle - neck_point
    cam_axis = cam_point - center_of_circle

    return 180 - get_angle(cam_axis, neck_axis)


def left_alpha_angle(points):
    return alpha_angle(points, 0)


def right_alpha_angle(points):
    return alpha_angle(points, 15)


def average_alpha_angle(points):
    left_aa = left_alpha_angle(points)
    right_aa = right_alpha_angle(points)
    return (left_aa + right_aa) / 2.0


def lce_angle(points, startIdx):
    center_of_circle = get_center_of_circle(points[startIdx + 10], points[startIdx + 11], points[startIdx + 12])
    center_of_circle = center_of_circle.to(points.device)
    lat_point = points[startIdx]

    lat_axis = lat_point - center_of_circle
    up_axis = torch.Tensor([0, -1]).to(points.device).float()

    return get_angle(lat_axis.float(), up_axis)


def left_lce_angle(points):
    return lce_angle(points, 0)


def right_lce_angle(points):
    return lce_angle(points, 15)


def neck_shaft_angle(points, startIdx):
    axis_1 = points[startIdx + 2] - points[startIdx + 5]
    axis_2 = points[startIdx + 5] - points[startIdx + 6]

    return get_angle(axis_1, axis_2)


def left_neck_shaft_angle(points):
    return neck_shaft_angle(points, 0)


def right_neck_shaft_angle(points):
    return neck_shaft_angle(points, 15)


def acetabular_index(points, side):
    startIdx = 0 if side == "left" else 15
    vector = [0, 1]

    axis_1 = points[startIdx + 1] - points[startIdx]
    horizontal_axis = torch.Tensor(vector).to(points.device).float()

    return get_angle(axis_1.float(), horizontal_axis)


def left_acetabular_index(points):
    return acetabular_index(points, "left")


def right_acetabular_index(points):
    return acetabular_index(points, "right")


def pelvic_tilt(points):

    pelvic_axis = points[9] - points[24]
    horizontal_axis = torch.Tensor([1, 0]).to(points.device).float()

    return get_angle(pelvic_axis.float(), horizontal_axis)