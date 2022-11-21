import torch

from lib.utils import get_angle
from lib.utils import get_center_of_circle


def alpha_angle(points, center_idx, neck_point_idx, cam_point_idx):
    center_of_circle = points[center_idx]
    neck_point = points[neck_point_idx]
    cam_point = points[cam_point_idx]

    neck_axis = center_of_circle - neck_point
    cam_axis = cam_point - center_of_circle

    return 180 - get_angle(cam_axis, neck_axis)


def left_alpha_angle(points):
    return alpha_angle(points, 1, 2, 6)


def right_alpha_angle(points):
    return alpha_angle(points, 4, 5, 7)


def average_alpha_angle(points):
    left_aa = left_alpha_angle(points)
    right_aa = right_alpha_angle(points)
    return (left_aa + right_aa) / 2.0


def lce_angle(points, lat_idx, center_idx):
    center_of_circle = points[center_idx]
    lat_point = points[lat_idx]

    lat_axis = lat_point - center_of_circle
    up_axis = torch.Tensor([0, -1]).to(points.device).float()

    return get_angle(lat_axis.float(), up_axis)


def left_lce_angle(points):
    return lce_angle(points, 3, 4)


def right_lce_angle(points):
    return lce_angle(points, 0, 1)


def fai(points):

    l_aa = left_alpha_angle(points)
    l_lce = left_lce_angle(points)

    r_aa = right_alpha_angle(points)
    r_lce = right_lce_angle(points)

    # diagnose left
    aa_threshold = 65
    lce_threshold = 40

    if l_aa < aa_threshold and l_lce < lce_threshold:
        l_fai = 0
    else:
        l_fai = 1

    if r_aa < aa_threshold and r_lce < lce_threshold:
        r_fai = 0
    else:
        r_fai = 1

    return [l_fai, r_fai]