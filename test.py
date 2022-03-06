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
    all_losses = []

    with torch.no_grad():
        for idx, (image, channels, meta) in enumerate(test_loader):

            output_per_model = []
            losses_per_model = []
            scaled_predicted_points_per_model = []
            target_points = None
            scaled_target_points = None
            eres_per_model = []

            for our_model in ensemble:
                output = our_model(image.float())
                output = two_d_softmax(output)
                output_per_model.append(output.cpu().detach().numpy())

                loss = nll_across_batch(output, channels)
                losses_per_model.append(loss)

                scaled_predicted_points, scaled_target_points, eres \
                    = get_predicted_and_target_points(output, meta['landmarks_per_annotator'], meta['pixel_size'])
                scaled_predicted_points_per_model.append(scaled_predicted_points.cpu().detach().numpy())
                target_points = torch.mean(meta['landmarks_per_annotator'], dim=1).cpu().detach().numpy()
                eres_per_model.append(eres.cpu().detach().numpy())

            recp_points = get_ere_reciprocal_weighted_points(np.array(scaled_predicted_points_per_model), np.array(eres_per_model))
            scaled_target_points = scaled_target_points.cpu().detach().numpy()

            displacement_vectors = recp_points - scaled_target_points
            radial_errors = np.linalg.norm(displacement_vectors, axis=2)

            # Print loss, radial error for each landmark and MRE for the image
            # Assumes that the batch size is 1 here
            msg = "Image: {}\tloss: {:.3f}".format(meta['file_name'][0], loss.item())
            for radial_error in radial_errors[0]:
                msg += "\t{:.3f}mm".format(radial_error)
            msg += "\taverage: {:.3f}mm".format(np.mean(radial_errors))
            logger.info(msg)

            # if np.any(radial_errors > 10):
            #     for keypoint_idx in np.argwhere(np.squeeze(radial_errors) > 10):
            # for keypoint_idx in range(cfg.DATASET.KEY_POINTS):
            visualise_ensemble_2(image.cpu().detach().numpy(),
                                 output_per_model,
                                 recp_points,
                                 target_points,
                                 eres_per_model)

    # Overall loss
    '''
    logger.info("Average loss: {:.3f}".format(np.mean(all_losses)))

    # Get 'all' statistics
    all_radial_errors = [val['radial_errors'] for val in data_tracker.values()]
    all_expected_radial_errors = [val['expected_radial_errors'] for val in data_tracker.values()]
    all_mode_probabilities = [val['mode_probabilities'] for val in data_tracker.values()]

    # MRE per landmark
    all_radial_errors = np.array(all_radial_errors)
    mre_per_landmark = np.mean(all_radial_errors, axis=0)
    msg = "Average radial error per landmark: "
    for mre in mre_per_landmark:
        msg += "\t{:.3f}mm".format(mre)
    logger.info(msg
    '''

    # Total MRE
    mre = np.mean(all_radial_errors)
    logger.info("Average radial error (MRE): {:.3f}mm".format(mre))

    # Detection rates
    flattened_radial_errors = all_radial_errors.flatten()
    '''
    sdr_statistics = produce_sdr_statistics(flattened_radial_errors, [2.0, 2.5, 3.0, 4.0])
    logger.info("Successful Detection Rate (SDR) for 2mm, 2.5mm, 3mm and 4mm respectively: "
                "{:.3f}% {:.3f}% {:.3f}% {:.3f}%".format(*sdr_statistics))
    '''
    sdr_statistics = produce_sdr_statistics(flattened_radial_errors, [2.0, 4.0, 10.0])
    logger.info("Successful Detection Rate (SDR) for 2mm, 4mm and 10mm respectively: "
                "{:.3f}% {:.3f}% {:.3f}%".format(*sdr_statistics))


if __name__ == '__main__':
    main()