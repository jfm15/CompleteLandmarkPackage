import argparse
import torch
import glob
import os

import lib

from lib.dataset.landmark_dataset import LandmarkDataset
from lib.utils import prepare_for_testing
from lib.core.validate import validate

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

    parser.add_argument('--visuals',
                        help='list of graphics: p = predictions, g = ground truth,'
                             'h = heatmaps, e = ere scores, a = aggregation',
                        nargs='+',
                        required=False,
                        default=[])

    parser.add_argument('--measurements',
                        help='list of functions to call in measurements',
                        nargs='+',
                        required=False,
                        default=[])

    args = parser.parse_args()

    return args


def main():

    # Get arguments and the experiment file
    args = parse_args()

    cfg, logger, output_path, yaml_file_name = prepare_for_testing(args.cfg, args.pretrained_model_directory)

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
        our_model = eval("lib.models." + cfg.MODEL.NAME)(cfg.MODEL, cfg.DATASET.KEY_POINTS)
        loaded_state_dict = torch.load(model_path)
        our_model.load_state_dict(loaded_state_dict, strict=True)
        our_model.eval()
        ensemble.append(our_model)

    logger.info("-----------Model Summary-----------")
    model_summary, _ = summary_string(ensemble[0], (1, *cfg.DATASET.CACHED_IMAGE_SIZE), device=torch.device('cpu'))
    logger.info(model_summary)

    image_save_path = os.path.join(output_path, '../images')
    if not os.path.exists(image_save_path):
        os.makedirs(image_save_path)

    # call the validate function
    validate(cfg, ensemble, args.testing_images, test_loaders, args.visuals, args.measurements,
             logger, print_progress=True, image_save_path=image_save_path)


if __name__ == '__main__':
    main()