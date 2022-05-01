import os
import time
import logging

from config import get_cfg_defaults


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

    alpha_angle_differences = []
    beta_angle_differences = []

    for i in range(pred_base_lines.shape[0]):
        pred_base_line = pred_base_lines[i]
        pred_bony_roof_line = pred_bony_roof_lines[i]
        pred_cartilage_roof_line = pred_cartilage_roof_lines[i]

        tar_base_line = tar_base_lines[i]
        tar_bony_roof_line = tar_bony_roof_lines[i]
        tar_cartilage_roof_line = tar_cartilage_roof_lines[i]

        print(i, pred_base_line, tar_base_line)

    return