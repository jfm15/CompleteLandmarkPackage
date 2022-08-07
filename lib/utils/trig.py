import torch


def get_angle(v1, v2):
    v1_mag = torch.norm(v1)
    v2_mag = torch.norm(v2)
    dot_product = torch.dot(v1, v2)
    angle = torch.acos(dot_product / (v1_mag * v2_mag))
    return torch.rad2deg(angle).item()


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