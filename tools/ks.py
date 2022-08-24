import argparse
import torch
import os
import random

import _init_paths
import lib
import json

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from lib.dataset import LandmarkDataset
from lib.utils import prepare_for_dataset_preperation

from lib.measures import measure


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

    parser.add_argument('--partition_directory',
                        help='The path to the directory where the partition should be saved',
                        type=str,
                        required=True,
                        default='')

    parser.add_argument('--measurement',
                        help='What measurement to use as a representation of each image',
                        type=str,
                        required=True,
                        default='')

    parser.add_argument('--split',
                        help='',
                        type=float,
                        nargs='+',
                        required=True,
                        default='')

    args = parser.parse_args()

    return args


def main():
    # get arguments and the experiment file
    args = parse_args()

    cfg, output_path = prepare_for_dataset_preperation(args.cfg)

    loaders = []
    for image_directory in args.images:
        dataset = LandmarkDataset(image_directory, args.annotations, cfg.DATASET)
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
        loaders.append(loader)

    # Create a dict like this:
    #
    #     dict = {
    #         {ID}: {Value}
    #         ...
    #     }
    #

    dict = {}

    for loader in loaders:

        for _, _, meta in loader:

            b = 0
            id = meta['file_name'][b]
            target_points = torch.mean(meta['landmarks_per_annotator'], dim=1)
            scaled_target_points = torch.multiply(target_points, meta['pixel_size'])

            true_value, _, _ = measure(scaled_target_points[b], scaled_target_points[b],
                                cfg.VALIDATION.MEASUREMENTS_SUFFIX, args.measurement)
            dict[id] = true_value

    keys = list(dict.keys())
    values = list(dict.values())
    no_annotation_paths = len(keys)
    target_no_in_each_set = np.floor(no_annotation_paths * np.array(args.split)).astype(int)
    cumulative_no_in_each_set = np.cumsum(target_no_in_each_set)
    cumulative_no_in_each_set[-1] = no_annotation_paths
    cumulative_no_in_each_set = np.insert(cumulative_no_in_each_set, 0, 0)

    partition_labels = ["training", "validation", "testing"]

    # create partition loop
    best_partition = {}
    best_ks_score = 1
    best_ys_for_each_label = {}

    xs = np.linspace(np.min(values) - 4, np.max(values) + 4, num=1000)
    for _ in tqdm(range(10 ** 5)):

        random.shuffle(keys)

        partition_ids = {}
        partition_values = {}
        for i, label in enumerate(partition_labels):
            partition_ids[label] = keys[cumulative_no_in_each_set[i]: cumulative_no_in_each_set[i + 1]]
            partition_values[label] = []
            for id in partition_ids[label]:
                partition_values[label].append(dict[id])
            partition_values[label] = np.sort(partition_values[label])

        # create a cdf
        ys_for_each_label = []
        for label in partition_labels:
            xp = partition_values[label]
            fp = np.linspace(0, 1, num=len(xp))
            ys = np.interp(xs, xp, fp)
            ys_for_each_label.append(ys)

        stack = np.stack(ys_for_each_label)
        ptp = np.ptp(stack, axis=0)

        ks_score = np.max(ptp)

        if ks_score < best_ks_score:
            best_ks_score = ks_score
            best_partition = partition_ids
            best_ys_for_each_label = ys_for_each_label

    # graph the KS
    stack = np.stack(best_ys_for_each_label)
    ptp = np.ptp(stack, axis=0)
    ks_score = np.max(ptp)
    ks_index = np.argmax(ptp)
    aa_index = xs[ks_index]
    print("KS score", ks_score)

    plt.figure(
        figsize=(6, 6))
    # plt.rcParams["font.weight"] = "bold"
    plt.rcParams["font.size"] = "12"
    plt.xlabel('Alpha Angle', fontsize=12)
    plt.ylabel('Cumulative Proportion of Dataset', fontsize=12)
    colors = ['r', 'g', 'b']
    for i, ys in enumerate(best_ys_for_each_label):
        plt.plot(xs, ys, (colors[i] + "-"))
    x = aa_index
    y = np.min(stack[:, ks_index])
    dx = 0
    dy = np.ptp(stack[:, ks_index])
    plt.plot([x, x], [y, y + dy], color='k')
    plt.text(x + dx - 20, y + dy, "KS test = {:.4f}".format(ks_score), fontsize=12)
    plt.grid(True)
    plt.show()

    partition_filename = "partition_{}_".format(args.measurement)
    for split in args.split:
        partition_filename += "{}_".format(split)
    partition_filename += "{:.5f}.json".format(best_ks_score)

    partition_save_path = os.path.join(args.partition_directory, partition_filename)
    print("Saving file to {}".format(partition_save_path))
    partition_file = open(partition_save_path, "w")
    json.dump(best_partition, partition_file)
    partition_file.close()


if __name__ == '__main__':
    main()