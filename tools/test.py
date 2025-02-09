import argparse
import torch
import glob
import os

import _init_paths
import lib
import lib.models

from lib.dataset import LandmarkDataset
from lib.utils import prepare_for_testing

import lib.core.validate_cpu as validate_cpu
import lib.core.validate_gpu as validate_gpu

from torchsummary.torchsummary import summary_string


def parse_args():
    parser = argparse.ArgumentParser(description='Test a network trained to detect landmarks')

    parser.add_argument('--cfg',
                        help='The path to the configuration file for the experiment',
                        required=True,
                        type=str)

    parser.add_argument('--images',
                        help='The path to the training images',
                        type=str,
                        required=True,
                        default='')

    parser.add_argument('--annotations',
                        help='The path to the directory where annotations are stored',
                        type=str,
                        required=True,
                        default='')

    parser.add_argument('--partition',
                        help='The path to the partition file',
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

    parser.add_argument('--validation',
                        action='store_true',
                        help='If this is given run over the validation set instead of the test set')

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

    partition_label = "validation" if args.validation else "testing"

    test_dataset = LandmarkDataset(args.images, args.annotations, cfg.DATASET, gaussian=False,
                                   perform_augmentation=False, partition=args.partition, partition_label=partition_label)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Load models and state dict from file
    ensemble = []
    model_paths = sorted(glob.glob(args.pretrained_model_directory + "/*.pth"))

    final_layer = eval("lib.models." + cfg.TRAIN.FINAL_LAYER)
    loss_function = eval("lib.models." + cfg.TRAIN.LOSS_FUNCTION)

    for model_path in model_paths:
        our_model = eval("lib.models." + cfg.MODEL.NAME)(cfg.MODEL, cfg.DATASET.KEY_POINTS)
        loaded_state_dict = torch.load(model_path)
        our_model.load_state_dict(loaded_state_dict, strict=True)
        our_model.eval()
        ensemble.append(our_model)

    logger.info("-----------Model Summary-----------")
    model_summary, _ = summary_string(ensemble[0], (1, *cfg.DATASET.CACHED_IMAGE_SIZE), device=torch.device('cpu'))
    logger.info(model_summary)

    image_save_path = os.path.join(output_path, 'images')
    if not os.path.exists(image_save_path):
        os.makedirs(image_save_path)

    # Validate
    with torch.no_grad():

        if torch.cuda.is_available():
            validate_file = "validate_gpu"
        else:
            validate_file = "validate_cpu"

        eval("{}.validate_over_set".format(validate_file)) \
            (ensemble, test_loader, final_layer, loss_function, args.visuals, cfg.VALIDATION, image_save_path,
             logger=logger)



if __name__ == '__main__':
    main()