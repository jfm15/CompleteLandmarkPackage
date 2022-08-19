import argparse
import torch

import _init_paths
import lib

from lib.dataset import LandmarkDataset
from lib.utils import prepare_config
from sklearn.decomposition import PCA

from scipy.spatial import procrustes
from scipy.linalg import orthogonal_procrustes

import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description='Test a network trained to detect landmarks')

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
                        nargs='+',
                        required=True,
                        default='')

    parser.add_argument('--annotations',
                        help='The path to the directory where annotations are stored',
                        type=str,
                        required=True,
                        default='')

    args = parser.parse_args()

    return args


# Make the points centered
# Divide by the sum of squares of every entry
def standardize(points):
    mean = torch.mean(points, dim=0, keepdim=False)
    centered_points = points - mean
    norm = torch.norm(centered_points)
    return centered_points / norm, mean, norm


def remap(points, base):

    standardized_base, base_mean, base_norm = standardize(base)
    standardized_points, _, _ = standardize(points)

    R, s = orthogonal_procrustes(standardized_base, standardized_points)
    standardized_remapped = torch.matmul(standardized_points, torch.from_numpy(R.T)) * s

    return base_mean + (standardized_remapped * base_norm)


def main():

    # Get arguments and the experiment file
    args = parse_args()

    cfg = prepare_config(args.cfg)

    training_dataset = LandmarkDataset(args.training_images, args.annotations, cfg.DATASET, gaussian=False,
                                   perform_augmentation=False)
    training_loader = torch.utils.data.DataLoader(training_dataset, batch_size=1, shuffle=False)

    validation_dataset = LandmarkDataset(args.validation_images[0], args.annotations, cfg.DATASET, gaussian=False,
                                       perform_augmentation=False)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=1, shuffle=False)

    # Get normalized mean
    all_target_points = []
    for idx, (image, _, meta) in enumerate(training_loader):
        target_points = torch.mean(meta['landmarks_per_annotator'], dim=1)[0]
        all_target_points.append(target_points)

    all_target_points = torch.stack(all_target_points)
    X_ = torch.mean(all_target_points, dim=0)

    difference_vectors = []
    for idx, (image, _, meta) in enumerate(training_loader):
        target_points = torch.mean(meta['landmarks_per_annotator'], dim=1)[0]
        remapped = remap(target_points, X_)
        difference_vectors.append(remapped - X_)

    difference_vectors = torch.stack(difference_vectors)
    difference_vectors = difference_vectors.reshape(difference_vectors.size()[0], -1)

    pca = PCA(n_components=difference_vectors.size()[1])
    pca.fit(difference_vectors)
    components = torch.from_numpy(pca.components_)[:10]

    for idx, (image, _, meta) in enumerate(validation_loader):
        # Convert landmarks to standardized space
        target_points = torch.mean(meta['landmarks_per_annotator'], dim=1)[0]

        # Add error
        predicted_points = target_points.clone()
        # predicted_points += torch.rand(target_points.size()) * 20 - 10
        predicted_points[0] += torch.Tensor([0, 100])

        remapped = remap(predicted_points, X_)
        difference_vector = (remapped - X_).reshape(1, -1)
        keys = torch.from_numpy(pca.transform(difference_vector)[0])[:10]
        processed_points = X_ + torch.reshape(torch.matmul(keys, components), (-1, 2))
        # This isn't the best, we want to use the inverse of the remap operation above
        processed_points = remap(processed_points, predicted_points)

        plt.imshow(image[0, 0], cmap='gray')
        plt.scatter(target_points[:, 0], target_points[:, 1], s=5)
        plt.scatter(predicted_points[:, 0], predicted_points[:, 1], s=5)
        plt.scatter(processed_points[:, 0], processed_points[:, 1], s=5)
        plt.show()

if __name__ == '__main__':
    main()
