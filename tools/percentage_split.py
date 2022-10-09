import argparse
import os
import glob

import _init_paths
import lib
import json
import random

import numpy as np

from lib.utils import prepare_for_dataset_preperation


'''
Code design based on Bin Xiao's Deep High Resolution Network Repository:
https://github.com/leoxiaobin/deep-high-resolution-net.pytorch
'''


def parse_args():
    parser = argparse.ArgumentParser(description='Train a network to detect landmarks')

    parser.add_argument('--annotations',
                        help='The path to the directory where annotations are stored',
                        type=str,
                        required=True,
                        default='')

    parser.add_argument('--partition_directory',
                        help='The path to the directory where the partition should be saved',
                        type=str,
                        required=True,
                        default='')

    parser.add_argument('--split',
                        help='',
                        type=float,
                        nargs='+',
                        required=True,
                        default='')

    parser.add_argument('--shuffle',
                        action='store_true',
                        help='shuffle examples before splitting')

    parser.add_argument('--n',
                        help='number of runs',
                        type=int,
                        required=False,
                        default=1)

    args = parser.parse_args()

    return args


def main():
    # get arguments and the experiment file
    args = parse_args()

    # Get sub-directories for annotations
    annotation_sub_dirs = sorted(glob.glob(args.annotations + "/*.txt"))

    no_annotation_paths = len(annotation_sub_dirs)
    target_no_in_each_set = np.floor(no_annotation_paths * np.array(args.split)).astype(int)
    cumulative_no_in_each_set = np.cumsum(target_no_in_each_set)
    cumulative_no_in_each_set[-1] = no_annotation_paths
    cumulative_no_in_each_set = np.insert(cumulative_no_in_each_set, 0, 0)

    for i in range(len(annotation_sub_dirs)):
        annotation_sub_dirs[i] = annotation_sub_dirs[i].split("/")[-1]
        annotation_sub_dirs[i] = annotation_sub_dirs[i].split(".")[0]

    partition_labels = ["training", "validation", "testing"]

    for run in range(args.n):

        if args.shuffle:
            random.shuffle(annotation_sub_dirs)

        partition_ids = {}

        for i, label in enumerate(partition_labels):
            partition_ids[label] = annotation_sub_dirs[cumulative_no_in_each_set[i]: cumulative_no_in_each_set[i + 1]]

        partition_filename = "partition"
        for split in args.split:
            partition_filename += "_{}".format(split)

        if args.shuffle:
            partition_filename += "_shuffled"

        if args.n > 1:
            partition_filename += "_{}".format(run)

        partition_filename += ".json"

        partition_save_path = os.path.join(args.partition_directory, partition_filename)
        print("Saving file to {}".format(partition_save_path))
        partition_file = open(partition_save_path, "w")
        json.dump(partition_ids, partition_file)
        partition_file.close()


if __name__ == '__main__':
    main()