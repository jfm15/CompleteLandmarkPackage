import os
import time
import logging

import torch
import numpy as np

from config import get_cfg_defaults
from sklearn.metrics import confusion_matrix


def prepare_for_training(cfg_path, output_path):
    cfg = prepare_config(cfg_path)

    # get directory to save log and model
    split_cfg_path = cfg_path.split("/")
    yaml_file_name = os.path.splitext(split_cfg_path[-1])[0]
    output_path = os.path.join(output_path, yaml_file_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    log_path = os.path.join(output_path, get_log_file_name('train'))

    # setup the logger
    logger = setup_logger(log_path)

    return cfg, logger, output_path, yaml_file_name


def prepare_for_testing(cfg_path, model_path):
    cfg = prepare_config(cfg_path)

    split_cfg_path = cfg_path.split("/")
    yaml_file_name = os.path.splitext(split_cfg_path[-1])[0]
    model_name = os.path.basename(model_path)
    output_path = os.path.join('output', yaml_file_name, model_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    log_path = os.path.join(output_path, get_log_file_name('test'))

    # setup the logger
    logger = setup_logger(log_path)

    return cfg, logger, output_path, yaml_file_name


def prepare_config(cfg_path):
    # get config
    cfg = get_cfg_defaults()
    cfg.merge_from_file(cfg_path)
    cfg.freeze()
    return cfg


def get_log_file_name(log_prefix):
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}.log'.format(log_prefix, time_str)
    return log_file


def setup_logger(log_path):
    logging.basicConfig(filename=log_path,
                        format='%(message)s')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)
    return logger


def get_validation_message(aggregated_point_mres, no_of_base_estimators, aggregation_methods,
                           sdr_method, sdr_thresholds, sdr_statistics):

    msg = "Average radial error per base model: "
    i = 0
    for _ in range(no_of_base_estimators):
        value = aggregated_point_mres[i]
        msg += "{:.3f}mm ".format(value)
        i += 1

    for aggregation_method in aggregation_methods:
        value = aggregated_point_mres[i]
        msg += "\nAverage radial error for {} aggregation: {:.3f}mm".format(aggregation_method, value)
        i += 1

    sdr_thresholds_formatted = ', '.join(["{}mm".format(threshold) for threshold in sdr_thresholds])
    sdr_statistics_formatted = ', '.join(["{:.3f}%".format(stat) for stat in sdr_statistics])
    msg += "\nThe Successful Detection Rate (SDR) for {} aggregation for thresholds {} respectively is {} "\
        .format(sdr_method, sdr_thresholds_formatted, sdr_statistics_formatted)

    return msg


def compare_angles(predicted_points, target_points):
    """

    :param predicted_points: torch.Size([no_of_images, 5, 2])
    :param target_points: torch.Size([no_of_images, 5, 2])
    :return:
    """

    pred_base_lines = predicted_points[:, 1, :] - predicted_points[:, 0, :]
    pred_bony_roof_lines = predicted_points[:, 3, :] - predicted_points[:, 2, :]
    pred_cartilage_roof_lines = predicted_points[:, 4, :] - predicted_points[:, 2, :]

    tar_base_lines = target_points[:, 1, :] - target_points[:, 0, :]
    tar_bony_roof_lines = target_points[:, 3, :] - target_points[:, 2, :]
    tar_cartilage_roof_lines = target_points[:, 4, :] - target_points[:, 2, :]

    predicted_alpha_angles = []
    tar_alpha_angles = []
    predicted_beta_angles = []
    tar_beta_angles = []
    predicted_diagnosis = []
    tar_diagnosis = []
    alpha_angle_differences = []
    beta_angle_differences = []

    for i in range(pred_base_lines.shape[0]):
        pred_base_line = pred_base_lines[i]
        pred_bony_roof_line = pred_bony_roof_lines[i]
        pred_cartilage_roof_line = pred_cartilage_roof_lines[i]

        tar_base_line = tar_base_lines[i]
        tar_bony_roof_line = tar_bony_roof_lines[i]
        tar_cartilage_roof_line = tar_cartilage_roof_lines[i]

        pred_alpha_angle = get_angle(pred_base_line, pred_bony_roof_line)
        predicted_alpha_angles.append(pred_alpha_angle)
        tar_alpha_angle = get_angle(tar_base_line, tar_bony_roof_line)
        tar_alpha_angles.append(tar_alpha_angle)
        alpha_angle_difference = abs(tar_alpha_angle - pred_alpha_angle)
        alpha_angle_differences.append(alpha_angle_difference)

        pred_beta_angle = get_angle(-pred_base_line, pred_cartilage_roof_line)
        predicted_beta_angles.append(pred_beta_angle)
        tar_beta_angle = get_angle(-tar_base_line, tar_cartilage_roof_line)
        tar_beta_angles.append(tar_beta_angle)
        beta_angle_difference = abs(tar_beta_angle - pred_beta_angle)
        beta_angle_differences.append(beta_angle_difference)

        # diagnose
        predicted_diagnosis.append(diagnose(pred_alpha_angle, pred_beta_angle))
        tar_diagnosis.append(diagnose(tar_alpha_angle, tar_beta_angle))

    predicted_alpha_angles = np.array(predicted_alpha_angles)
    tar_alpha_angles = np.array(tar_alpha_angles)
    predicted_beta_angles = np.array(predicted_beta_angles)
    tar_beta_angles = np.array(tar_beta_angles)

    alpha_angle_icc = get_icc(predicted_alpha_angles, tar_alpha_angles)
    beta_angle_icc = get_icc(predicted_beta_angles, tar_beta_angles)

    total = 0
    count = 0
    for pred_diag, tar_diag in zip(predicted_diagnosis, tar_diagnosis):
        if pred_diag == tar_diag:
            count += 1
        total += 1

    diagnosis_accuracy = 100 * float(count) / float(total)

    print(confusion_matrix(predicted_diagnosis, tar_diagnosis, labels=["1", "2a/b", "2c", "D", "3/4"]))

    return torch.Tensor(alpha_angle_differences), torch.Tensor(beta_angle_differences), \
           alpha_angle_icc, beta_angle_icc, diagnosis_accuracy


def get_angle(v1, v2):
    v1_mag = torch.norm(v1)
    v2_mag = torch.norm(v2)
    dot_product = torch.dot(v1, v2)
    angle = torch.acos(dot_product / (v1_mag * v2_mag))
    return torch.rad2deg(angle).item()


def get_icc(predictions, targets):
    # formula for ICC
    N = len(predictions)
    x_ = (np.sum(predictions) + np.sum(targets)) / (2.0 * N)
    s_2 = (np.sum(np.power(predictions - x_, 2)) + np.sum(np.power(targets - x_, 2))) / (2.0 * N)
    icc = np.sum((predictions - x_) * (targets - x_)) / (N * s_2)
    return icc


def diagnose(alpha, beta):

    if alpha >= 60:
        return "1"
    elif 50 <= alpha <= 59:
        return "2a/b"
    elif 43 <= alpha <= 49 and beta < 77:
        return "2c"
    elif 43 <= alpha <= 49 and beta > 77:
        return "D"
    elif alpha < 43:
        return "3/4"
