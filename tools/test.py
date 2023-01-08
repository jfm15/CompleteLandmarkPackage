import argparse
import torch
import wandb
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

    parser.add_argument('--tags',
                        help='tags which are passed to weights and biases',
                        nargs='+',
                        required=False,
                        default=[])

    parser.add_argument('--proposed_threshold',
                        help='The proposed threshold',
                        type=float,
                        required=False,
                        default=None)

    args = parser.parse_args()

    return args


def main():

    # Get arguments and the experiment file
    args = parse_args()

    cfg, logger, output_path, yaml_file_name = prepare_for_testing(args.cfg, args.pretrained_model_directory)

    wandb.login(key="f6e720fe9b2f70bdd25b65e68e51d5163e2b0337")

    tags = ['test'] + args.tags
    if cfg.TRAIN.ENSEMBLE_MODELS > 1:
        tags.append("ensemble")
    wandb.init(project="complete_landmark_package", name=yaml_file_name, config=cfg,
               entity="j-mccouat", tags=tags)

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

        loss_dict, mre_dict, _ = eval("{}.validate_over_set".format(validate_file)) \
            (ensemble, test_loader, final_layer, loss_function, args.visuals, cfg.VALIDATION, image_save_path,
             proposed_threhold=args.proposed_threshold, logger=logger)

        if cfg.TRAIN.ENSEMBLE_MODELS == 1:
            wandb.run.summary["loss"] = loss_dict["1"]
            wandb.run.summary["MRE"] = mre_dict["1"]
        else:
            # Add validation losses
            for model_idx, model_loss in loss_dict.items():
                wandb.run.summary["loss_{}".format(model_idx)] = model_loss

            # Add validation mre
            for model_idx, model_mre in mre_dict.items():
                wandb.run.summary["mre_{}".format(model_idx)] = model_mre


if __name__ == '__main__':
    main()