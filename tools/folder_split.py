import argparse
import os
import glob

import _init_paths
import lib
import json

import numpy as np

from lib.utils import prepare_for_dataset_preperation


'''
Code design based on Bin Xiao's Deep High Resolution Network Repository:
https://github.com/leoxiaobin/deep-high-resolution-net.pytorch
'''


def parse_args():
    parser = argparse.ArgumentParser(description='Train a network to detect landmarks')

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

    parser.add_argument('--partition_directory',
                        help='The path to the directory where the partition should be saved',
                        type=str,
                        required=True,
                        default='')

    args = parser.parse_args()

    return args


def main():
    # get arguments and the experiment file
    args = parse_args()

    # Get sub-directories for annotations
    training_images = sorted(glob.glob(args.training_images + "/*"))
    validation_images = sorted(glob.glob(args.validation_images + "/*"))
    testing_images = sorted(glob.glob(args.testing_images + "/*"))

    for image_paths in [training_images, validation_images, testing_images]:
        for i in range(len(image_paths)):
            image_paths[i] = image_paths[i].split("/")[-1]
            image_paths[i] = image_paths[i].split(".")[0]

    partition_ids = {
        "training": training_images,
        "validation": validation_images,
        "testing": testing_images
    }

    partition_filename = "partition"
    for part in ["training", "validation", "testing"]:
        partition_filename += "_{}".format(len(partition_ids[part]))
    partition_filename += ".json"

    partition_save_path = os.path.join(args.partition_directory, partition_filename)
    print("Saving file to {}".format(partition_save_path))
    partition_file = open(partition_save_path, "w")
    json.dump(partition_ids, partition_file)
    partition_file.close()


if __name__ == '__main__':
    main()