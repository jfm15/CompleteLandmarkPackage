import argparse
import torch
import glob

import model
from landmark_dataset import LandmarkDataset
from utils import prepare_for_testing
from utils import get_validation_message
from function import validate_ensemble
from evaluate import cal_radial_errors
from evaluate import use_aggregate_methods
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
    predicted_points_per_model, eres_per_model, target_points = \
        validate_ensemble(ensemble, test_loader, print_progress=True, logger=logger)

    aggregated_point_dict = use_aggregate_methods(predicted_points_per_model, eres_per_model,
                                                  aggregate_methods=cfg.VALIDATION.AGGREGATION_METHODS)
    aggregated_point_mres = [cal_radial_errors(predicted_points, target_points, mean=True) for
                             predicted_points in aggregated_point_dict.values()]

    chosen_radial_errors = cal_radial_errors(aggregated_point_dict[cfg.VALIDATION.SDR_AGGREGATION_METHOD],
                                             target_points)
    sdr_statistics = get_sdr_statistics(chosen_radial_errors, cfg.VALIDATION.SDR_THRESHOLDS)

    logger.info('-----------Overall Statistics-----------')
    msg = get_validation_message(aggregated_point_mres, cfg.TRAIN.ENSEMBLE_MODELS, cfg.VALIDATION.AGGREGATION_METHODS,
                                 cfg.VALIDATION.SDR_AGGREGATION_METHOD, cfg.VALIDATION.SDR_THRESHOLDS, sdr_statistics)
    logger.info(msg)


if __name__ == '__main__':
    main()