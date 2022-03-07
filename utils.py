import os
import time
import logging
import torch
import numpy as np

from config import get_cfg_defaults
from evaluate import get_predicted_and_target_points
from evaluate import get_hottest_points
from backup.evaluate import produce_sdr_statistics


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

    return cfg, logger, output_path


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


def train_model(model, final_layer, optimizer, scheduler, loader, loss_function, logger):
    model.train()
    losses_per_epoch = []

    for batch, (image, channels, meta) in enumerate(loader):

        # Put image and channels onto gpu
        image = image.cuda()
        channels = channels.cuda()

        output = model(image.float())
        output = final_layer(output)

        optimizer.zero_grad()
        loss = loss_function(output, channels)
        loss.backward()

        optimizer.step()

        losses_per_epoch.append(loss.item())

        if (batch + 1) % 5 == 0:
            logger.info("[{}/{}]\tLoss: {:.3f}".format(batch + 1, len(loader), np.mean(losses_per_epoch)))

    scheduler.step()


def evaluate_model(model, final_layer, loader, loss_function):
    model.eval()
    all_losses = []
    all_radial_errors = []
    all_eres = []

    with torch.no_grad():
        for idx, (image, channels, meta) in enumerate(loader):
            # Put image and channels onto gpu
            image = image.cuda()
            channels = channels.cuda()
            meta['landmarks_per_annotator'] = meta['landmarks_per_annotator'].cuda()
            meta['pixel_size'] = meta['pixel_size'].cuda()

            output = model(image.float())
            output = final_layer(output)

            loss = loss_function(output, channels)
            all_losses.append(loss.item())

            radial_errors, eres = evaluate_gpu(output, meta['landmarks_per_annotator'], meta['pixel_size'])
            all_radial_errors.extend(radial_errors.cpu().detach().numpy())
            all_eres.extend(eres.cpu().detach().numpy())

    model.cpu()

    return all_losses, all_radial_errors, all_eres


def use_model(model, final_layer, loader, loss_function, logger=None, print_per_image=False):
    model.eval()
    all_losses = []
    all_predicted_points = []
    all_predicted_pixel_points = []
    all_target_points = []
    all_eres = []
    file_names = []

    with torch.no_grad():
        for idx, (image, channels, meta) in enumerate(loader):
            # Put image and channels onto gpu
            image = image.cuda()
            channels = channels.cuda()
            meta['landmarks_per_annotator'] = meta['landmarks_per_annotator'].cuda()
            meta['pixel_size'] = meta['pixel_size'].cuda()
            file_names.extend(meta["file_name"])

            output = model(image.float())
            output = final_layer(output)

            loss = loss_function(output, channels)
            all_losses.append(loss.item())

            predicted_points, target_points, eres \
                = get_predicted_and_target_points(output, meta['landmarks_per_annotator'], meta['pixel_size'])
            all_predicted_points.append(predicted_points.cpu().detach().numpy())
            all_target_points.append(target_points.cpu().detach().numpy())
            all_eres.extend(eres.cpu().detach().numpy())

            predicted_pixel_points = get_hottest_points(output).cpu().detach().numpy()
            all_predicted_pixel_points.append(predicted_pixel_points)

            if print_per_image:
                radial_errors = cal_radial_errors(predicted_points.cpu().detach().numpy(),target_points.cpu().detach().numpy())

                msg = "Image: {}\tloss: {:.3f}".format(meta['file_name'][0], loss.item())
                for radial_error in radial_errors:
                    msg += "\t{:.3f}mm".format(radial_error)
                msg += "\taverage: {:.3f}mm".format(np.mean(radial_errors))
                logger.info(msg)

    model.cpu()

    return all_losses, all_predicted_points, all_target_points, all_eres, all_predicted_pixel_points, file_names


def cal_radial_errors(predicted_points, target_points):
    displacement_vectors = predicted_points - target_points
    return np.linalg.norm(displacement_vectors, axis=2).flatten()


def get_ere_sum_weighted_points(predicted_points_per_model, eres_per_model):
    sum_eres = np.sum(eres_per_model, axis=0, keepdims=True)
    inverted_eres = sum_eres - eres_per_model
    inverted_eres = np.expand_dims(inverted_eres, axis=3)
    weighted_points = np.multiply(inverted_eres, predicted_points_per_model)
    return np.sum(weighted_points, axis=0) / np.sum(inverted_eres, axis=0)


def get_ere_max_weighted_points(predicted_points_per_model, eres_per_model):
    max_eres = np.max(eres_per_model, axis=0, keepdims=True)
    inverted_eres = max_eres - eres_per_model
    inverted_eres = np.expand_dims(inverted_eres, axis=3)
    weighted_points = np.multiply(inverted_eres, predicted_points_per_model)
    return np.sum(weighted_points, axis=0) / np.sum(inverted_eres, axis=0)


def get_ere_reciprocal_weighted_points(predicted_points_per_model, eres_per_model):
    inverted_eres = np.reciprocal(eres_per_model)
    inverted_eres = np.expand_dims(inverted_eres, axis=3)
    weighted_points = np.multiply(inverted_eres, predicted_points_per_model)
    return np.sum(weighted_points, axis=0) / np.sum(inverted_eres, axis=0)


def get_least_ere_points(predicted_points_per_model, eres_per_model):
    least_ere_indices = np.argmin(eres_per_model, axis=0)
    grid = np.indices(least_ere_indices.shape)
    return predicted_points_per_model[least_ere_indices, grid[0], grid[1]]


def get_validation_message(predicted_points_per_model, eres_per_model, target_points, sdr_thresholds):

    # Get radial errors for each model
    avg_radial_errors = [np.mean(cal_radial_errors(predicted_points, target_points))
                         for predicted_points in predicted_points_per_model]

    # Get radial error by averaging the models
    mean_model_points = np.mean(predicted_points_per_model, axis=0)
    mean_model_radial_errors = cal_radial_errors(mean_model_points, target_points)
    avg_radial_errors.append(np.mean(mean_model_radial_errors))

    # Get radial error using reciprocal ere method
    rec_weighted_model_points = get_ere_reciprocal_weighted_points(predicted_points_per_model, eres_per_model)
    rec_weighted_model_radial_errors = cal_radial_errors(rec_weighted_model_points, target_points)
    avg_radial_errors.append(np.mean(rec_weighted_model_radial_errors))

    # Print loss, radial error for each landmark and MRE for the image
    # Assumes that the batch size is 1 here
    msg = " Avg Radial Error per model: {:.3f}mm {:.3f}mm {:.3f}mm Avg Aggregation: {:.3f}mm " \
          "Confidence Weighted Aggregation: {:.3f}mm \\" \
        .format(*avg_radial_errors)

    sdr_statistics = produce_sdr_statistics(rec_weighted_model_radial_errors, sdr_thresholds)
    sdr_thresholds_formatted = ', '.join([str(threshold) + "mm" for threshold in sdr_thresholds])
    sdr_statistics_formatted = ', '.join(["{:.3f}%".format(stat) for stat in sdr_statistics])
    msg += "Successful Detection Rate (SDR) for {} respectively are: {} "\
        .format(sdr_thresholds_formatted, sdr_statistics_formatted)

    return msg