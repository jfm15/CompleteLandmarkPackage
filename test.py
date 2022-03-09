import argparse
import torch
import glob
import os

import numpy as np

import model
from model import two_d_softmax
from model import nll_across_batch
from landmark_dataset import LandmarkDataset
from utils import prepare_for_testing
from utils import get_validation_message
from function import use_model
from evaluate import cal_radial_errors
from evaluate import get_confidence_weighted_points
from evaluate import get_sdr_statistics
from torchsummary.torchsummary import summary_string


def parse_args():
    parser = argparse.ArgumentParser(description='Test a network trained to detect landmarks')

    parser.add_argument('--cfg',
                        help='The path to the configuration file for the experiment',
                        required=True,
                        type=str)

    parser.add_argument('--testing_images',
                        help='The path to the testing images',
                        type=str,
                        required=True,
                        default='')

    parser.add_argument('--annotations',
                        help='The path to the directory where annotations are stored',
                        type=str,
                        required=True,
                        default='')

    parser.add_argument('--pretrained_model_directory',
                        help='the path to a pretrained models',
                        type=str,
                        required=True)

    args = parser.parse_args()

    return args


def main():

    # Get arguments and the experiment file
    args = parse_args()

    cfg, logger, output_path, yaml_file_name = prepare_for_testing(args.cfg, args.pretrained_model_directory,
                                                                   args.testing_images)

    # Print the arguments into the log
    logger.info("-----------Arguments-----------")
    logger.info(vars(args))
    logger.info("")

    # Print the configuration into the log
    logger.info("-----------Configuration-----------")
    logger.info(cfg)
    logger.info("")

    # Load the testing dataset and put it into a loader
    test_dataset = LandmarkDataset(args.testing_images, args.annotations, cfg.DATASET, perform_augmentation=False)
    #test_dataset.set_specific_image("cache/cephalometric/640_800/150.png")
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Load models and state dict from file
    ensemble = []
    model_paths = sorted(glob.glob(args.pretrained_model_directory + "/*.pth"))

    for model_path in model_paths:
        our_model = eval("model." + cfg.MODEL.NAME)(cfg.MODEL, cfg.DATASET.KEY_POINTS)
        loaded_state_dict = torch.load(model_path)
        our_model.load_state_dict(loaded_state_dict, strict=True)
        our_model.eval()
        ensemble.append(our_model)

    logger.info("-----------Model Summary-----------")
    model_summary, _ = summary_string(ensemble[0], (1, *cfg.DATASET.CACHED_IMAGE_SIZE), device=torch.device('cpu'))
    logger.info(model_summary)

    logger.info("-----------Start Testing-----------")
    avg_loss_per_model = []
    predicted_points_per_model = []
    eres_per_model = []
    target_points = None
    save_image_path = os.path.join(cfg.VALIDATION.SAVE_IMAGE_PATH, yaml_file_name)
    if not os.path.exists(save_image_path):
        os.makedirs(save_image_path)

    for model_idx in range(len(ensemble)):
        logger.info('-----------Running Model {}-----------'.format(model_idx))
        our_model = ensemble[model_idx]
        our_model = our_model.cuda()

        all_losses, all_predicted_points, target_points, all_eres \
            = use_model(our_model, two_d_softmax, test_loader, nll_across_batch,
                        logger=logger, print_progress=True)

        predicted_points_per_model.append(all_predicted_points)
        eres_per_model.append(all_eres)

        # radial_error_per_model.append(all_radial_errors)
        avg_loss_per_model.append(all_losses)

        # move model back to cpu
        our_model.cpu()

    predicted_points_per_model = torch.stack(predicted_points_per_model)
    eres_per_model = torch.stack(eres_per_model)
    # predicted_points_per_model is size [M, D, N, 2]
    # eres_per_model is size [M, D, N]
    # target_points is size [D, N, 2]

    # perform analysis
    per_model_mre = [cal_radial_errors(predicted_points, target_points, mean=True)
                     for predicted_points in predicted_points_per_model]

    mean_aggregation_points = torch.mean(predicted_points_per_model, dim=0)
    mean_aggregation_mre = cal_radial_errors(mean_aggregation_points, target_points, mean=True)

    confidence_weighted_points = get_confidence_weighted_points(predicted_points_per_model, eres_per_model)
    confidence_weighted_errors = cal_radial_errors(confidence_weighted_points, target_points)
    confidence_weighted_mre = torch.mean(confidence_weighted_errors).item()

    sdr_statistics = get_sdr_statistics(confidence_weighted_errors, cfg.VALIDATION.SDR_THRESHOLDS)

    logger.info('-----------Overall Statistics-----------')
    msg = get_validation_message(per_model_mre, mean_aggregation_mre, confidence_weighted_mre,
                                 cfg.VALIDATION.SDR_THRESHOLDS, sdr_statistics)
    logger.info(msg)


if __name__ == '__main__':
    main()