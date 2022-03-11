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
from evaluate import combined_test_results
from torchsummary.torchsummary import summary_string


def parse_args():
    parser = argparse.ArgumentParser(description='Test a network trained to detect landmarks')

    parser.add_argument('--cfg',
                        help='The path to the configuration file for the experiment',
                        required=True,
                        type=str)

    parser.add_argument('--testing_images',
                        help='The path to the validation images',
                        type=str,
                        nargs='+',
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

    test_datasets = []
    test_loaders = []
    for test_images_path in args.testing_images:
        test_dataset = LandmarkDataset(test_images_path, args.annotations, cfg.DATASET, gaussian=False,
                                         perform_augmentation=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
        test_datasets.append(test_dataset)
        test_loaders.append(test_loader)

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

    aggregated_mres_per_test_set = []
    sdr_statistics_per_test_set = []
    no_per_test_set = []
    for i, testing_images_path in enumerate(args.testing_images):
        logger.info("\n-----------Testing over {}-----------".format(testing_images_path))
        predicted_points_per_model, eres_per_model, target_points = \
            validate_ensemble(ensemble, test_loaders[i], print_progress=True, logger=logger)
        no_per_test_set.append(len(test_loaders[i]))

        aggregated_point_dict = use_aggregate_methods(predicted_points_per_model, eres_per_model,
                                                      aggregate_methods=cfg.VALIDATION.AGGREGATION_METHODS)
        aggregated_point_mres = [cal_radial_errors(predicted_points, target_points, mean=True) for
                                 predicted_points in aggregated_point_dict.values()]
        aggregated_mres_per_test_set.append(aggregated_point_mres)

        chosen_radial_errors = cal_radial_errors(aggregated_point_dict[cfg.VALIDATION.SDR_AGGREGATION_METHOD],
                                                 target_points)
        sdr_statistics = get_sdr_statistics(chosen_radial_errors, cfg.VALIDATION.SDR_THRESHOLDS)
        sdr_statistics_per_test_set.append(sdr_statistics)

        logger.info('\n-----------Statistics for {}-----------'.format(testing_images_path))
        msg = get_validation_message(aggregated_point_mres, cfg.TRAIN.ENSEMBLE_MODELS, cfg.VALIDATION.AGGREGATION_METHODS,
                                     cfg.VALIDATION.SDR_AGGREGATION_METHOD, cfg.VALIDATION.SDR_THRESHOLDS, sdr_statistics)
        logger.info(msg)

    combined_aggregated_mres, combined_sdr_statistics = combined_test_results(aggregated_mres_per_test_set,
                                                                              sdr_statistics_per_test_set,
                                                                              no_per_test_set)

    if len(args.testing_images) > 1:
        logger.info('\n-----------Combined Statistics-----------')
        msg = get_validation_message(combined_aggregated_mres, cfg.TRAIN.ENSEMBLE_MODELS, cfg.VALIDATION.AGGREGATION_METHODS,
                                     cfg.VALIDATION.SDR_AGGREGATION_METHOD, cfg.VALIDATION.SDR_THRESHOLDS,
                                     sdr_statistics_per_test_set)
        logger.info(msg)


if __name__ == '__main__':
    main()