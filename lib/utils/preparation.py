import os
import time
import logging

import torch
import numpy as np

from lib.config.default import get_cfg_defaults
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


def prepare_for_dataset_preperation(cfg_path):
    cfg = prepare_config(cfg_path)

    split_cfg_path = cfg_path.split("/")
    yaml_file_name = os.path.splitext(split_cfg_path[-1])[0]

    output_path = os.path.join('output', yaml_file_name, "dataset_representation")

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    return cfg, output_path


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
