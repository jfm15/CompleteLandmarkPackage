import argparse
import torch
import json

import _init_paths
import lib

from lib.dataset import LandmarkDataset
from lib.utils import prepare_for_dataset_preperation

from lib.measures import measure
from lib.visualisations import display_measurement_distribution
from lib.visualisations import display_ks_score_of_partition


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

    parser.add_argument('--images',
                        help='The path to the images',
                        type=str,
                        required=True,
                        default='')

    parser.add_argument('--annotations',
                        help='The path to the directory where annotations are stored',
                        type=str,
                        required=True,
                        default='')

    parser.add_argument('--partition',
                        help='The partition to test',
                        type=str,
                        required=True,
                        default='')

    parser.add_argument('--measurement',
                        help='What measurement to use as a representation of each image',
                        type=str,
                        required=True,
                        default='')

    args = parser.parse_args()

    return args


def main():
    # get arguments and the experiment file
    args = parse_args()

    cfg, output_path = prepare_for_dataset_preperation(args.cfg)

    training_dataset = LandmarkDataset(args.images, args.annotations, cfg.DATASET, partition=args.partition,
                                       partition_label="training")
    training_loader = torch.utils.data.DataLoader(training_dataset, batch_size=1, shuffle=True)

    validation_dataset = LandmarkDataset(args.images, args.annotations, cfg.DATASET, gaussian=False,
                                         partition=args.partition, partition_label="validation")
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=1, shuffle=False)

    testing_dataset = LandmarkDataset(args.images, args.annotations, cfg.DATASET, gaussian=False,
                                      partition=args.partition, partition_label="testing")
    testing_loader = torch.utils.data.DataLoader(testing_dataset, batch_size=1, shuffle=False)

    # Create list for each measurement
    dict = {}

    for name, loader, color in zip(['training', 'validation', 'testing'],
                                   [training_loader, validation_loader, testing_loader],
                                   ['blue', 'green', 'red']):

        for _, _, meta in loader:
            b = 0
            id = meta['file_name'][b]

            target_points = torch.mean(meta['landmarks_per_annotator'], dim=1)
            scaled_target_points = torch.multiply(target_points, meta['pixel_size'])

            true_value, _, _ = measure(scaled_target_points[b], scaled_target_points[b],
                                cfg.VALIDATION.MEASUREMENTS_SUFFIX, args.measurement)
            dict[id] = true_value

        display_measurement_distribution(dict.values(), args.measurement, color)

    partition_file = open(args.partition)
    partition_dict = json.load(partition_file)

    display_ks_score_of_partition(partition_dict, dict, "statistic")


if __name__ == '__main__':
    main()