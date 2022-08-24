import argparse
import torch
import os

import _init_paths
import lib

from lib.dataset import LandmarkDataset
from lib.utils import prepare_for_dataset_preperation

from lib.measures import measure
from lib.visualisations import display_measurement_distribution


'''
Code design based on Bin Xiao's Deep High Resolution Network Repository:
https://github.com/leoxiaobin/deep-high-resolution-net.pytorch
'''


def parse_args():
    parser = argparse.ArgumentParser(description='Train a network to detect landmarks')

    parser.add_argument('--cfg',
                        help='The path to the configuration file for the experiment',
                        required=True,
                        type=str)

    parser.add_argument('--training_images',
                        help='The path to the training images',
                        type=str,
                        required=True,
                        default='')

    parser.add_argument('--validation_images',
                        help='The path to the validation images',
                        type=str,
                        required=True,
                        default='')

    parser.add_argument('--testing_images',
                        help='The path to the validation images',
                        type=str,
                        required=True,
                        default='')

    parser.add_argument('--annotations',
                        help='The path to the directory where annotations are stored',
                        type=str,
                        required=True,
                        default='')

    args = parser.parse_args()

    return args


def main():
    # get arguments and the experiment file
    args = parse_args()

    cfg, output_path = prepare_for_dataset_preperation(args.cfg)

    training_dataset = LandmarkDataset(args.training_images, args.annotations, cfg.DATASET)
    training_loader = torch.utils.data.DataLoader(training_dataset, batch_size=1, shuffle=False)

    validation_dataset = LandmarkDataset(args.validation_images, args.annotations, cfg.DATASET)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=1, shuffle=False)

    testing_dataset = LandmarkDataset(args.testing_images, args.annotations, cfg.DATASET)
    testing_loader = torch.utils.data.DataLoader(testing_dataset, batch_size=1, shuffle=False)

    for name, loader, color in zip(['training', 'validation', 'test'],
                                   [training_loader, validation_loader, testing_loader],
                                   ['blue', 'green', 'red']):

        # Create list for each measurement
        measurements_dict = {}
        for measurement in cfg.VALIDATION.MEASUREMENTS:
            measurements_dict[measurement] = []

        for _, _, meta in loader:

            target_points = torch.mean(meta['landmarks_per_annotator'], dim=1)
            scaled_target_points = torch.multiply(target_points, meta['pixel_size'])

            b = 0
            for measurement in cfg.VALIDATION.MEASUREMENTS:
                true_value, _, _ = measure(scaled_target_points[b], scaled_target_points[b],
                                    cfg.VALIDATION.MEASUREMENTS_SUFFIX, measurement)
                measurements_dict[measurement].append(true_value)

        for measurement in cfg.VALIDATION.MEASUREMENTS:
            save_path = os.path.join(output_path, "{}_{}_distribution".format(name, measurement))
            print("Saving to {}".format(save_path))
            display_measurement_distribution(measurements_dict[measurement], measurement, color, save_path)


if __name__ == '__main__':
    main()