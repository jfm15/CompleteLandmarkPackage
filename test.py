import argparse
import torch
import glob

import numpy as np
import matplotlib.pyplot as plt

import model
from model import two_d_softmax
from model import nll_across_batch
from visualise import visualise_ensemble_2
from evaluate import get_predicted_and_target_points
from utils import get_ere_reciprocal_weighted_points
from backup.evaluate import produce_sdr_statistics
from landmark_dataset import LandmarkDataset
from utils import prepare_for_testing
from utils import use_model
from utils import get_validation_message
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

    cfg, logger, output_path = prepare_for_testing(args.cfg, args.pretrained_model_directory, args.testing_images)

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
    model_paths = sorted(glob.glob(args.pretrained_model_directory + "*.pth"))

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

    for model_idx in range(len(ensemble)):
        our_model = ensemble[model_idx]
        our_model = our_model.cuda()

        all_losses, all_predicted_points, all_target_points, all_eres, _, _ \
            = use_model(our_model, two_d_softmax, test_loader, nll_across_batch)

        predicted_points_per_model.append(all_predicted_points)
        eres_per_model.append(all_eres)
        target_points = np.array(all_target_points)

        # radial_error_per_model.append(all_radial_errors)
        avg_loss_per_model.append(all_losses)

        # move model back to cpu
        our_model.cpu()

    predicted_points_per_model = np.array(predicted_points_per_model).squeeze()
    eres_per_model = np.array(eres_per_model).squeeze()
    target_points = np.squeeze(target_points)

    msg = get_validation_message(predicted_points_per_model, eres_per_model, target_points,
                                 cfg.VALIDATION.SDR_THRESHOLDS)

    logger.info(msg)


if __name__ == '__main__':
    main()