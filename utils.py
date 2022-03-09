import os
import time
import logging
import torch
import numpy as np

from config import get_cfg_defaults
from backup.evaluate import produce_sdr_statistics
from visualise import visualise_aggregations


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


def prepare_for_testing(cfg_path, model_path, data_path):
    cfg = prepare_config(cfg_path)

    split_cfg_path = cfg_path.split("/")
    yaml_file_name = os.path.splitext(split_cfg_path[-1])[0]
    model_name = os.path.basename(model_path)[:-len(".pth")]
    data_folder = os.path.basename(data_path)
    model_and_data_id = "{}_{}".format(model_name, data_folder)
    output_path = os.path.join('output', yaml_file_name, model_and_data_id)
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


def get_validation_message(logger, per_model_mre, mean_aggregation_mre, confidence_weighted_mre,
                           sdr_thresholds, sdr_statistics):

    # Print loss, radial error for each landmark and MRE for the image
    # Assumes that the batch size is 1 here
    per_model_mre_formatted = ', '.join(["{:.3f}mm".format(model_mre) for model_mre in per_model_mre])
    logger.info('-----------Overall Statistics-----------')
    msg = "Avg Radial Error per model: {} Mean Average Aggregation: {:.3f}mm " \
          "Confidence Weighted Aggregation: {:.3f}mm \n" \
        .format(per_model_mre_formatted, mean_aggregation_mre, confidence_weighted_mre)

    sdr_thresholds_formatted = ', '.join(["{}mm".format(threshold) for threshold in sdr_thresholds])
    sdr_statistics_formatted = ', '.join(["{:.3f}%".format(stat) for stat in sdr_statistics])
    msg += "Successful Detection Rate (SDR) for {} respectively are: {} "\
        .format(sdr_thresholds_formatted, sdr_statistics_formatted)

    return msg