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


def scale_matrix(scale):
    return torch.Tensor([[scale, 0], [0, scale]])


# Make the points centered
# Divide by the sum of squares of every entry
def standardize(points):
    mean = torch.mean(points, dim=0, keepdim=False)
    centered_points = points - mean
    norm = torch.norm(centered_points)
    return centered_points / norm, mean, norm


# Gets the best affine transformation which maps points onto the base
def getT(points, base):
    R, S = orthogonal_procrustes(points, base)
    T = torch.matmul(torch.from_numpy(R).float(), scale_matrix(S).float())
    return T.double()


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
    standardized_target_points = []
    for idx, (image, _, meta) in enumerate(training_loader):
        target_points = torch.mean(meta['landmarks_per_annotator'], dim=1)[0]
        standardized, _, _ = standardize(target_points)
        standardized_target_points.append(standardized)

    standardized_target_points = torch.stack(standardized_target_points)
    X_ = torch.mean(standardized_target_points, dim=0)

    difference_vectors = []
    for standardized in standardized_target_points:
        T = getT(X_, standardized)
        remapped = torch.matmul(standardized, T)
        difference_vectors.append(remapped - X_)

    difference_vectors = torch.stack(difference_vectors)
    difference_vectors = torch.reshape(difference_vectors, (difference_vectors.size()[0], -1))

    pca = PCA(n_components=difference_vectors.size()[1])
    pca.fit(difference_vectors)

    # get sigma_K
    bs = []
    for difference_vector in difference_vectors:
        difference_vector = torch.unsqueeze(difference_vector, 0)
        b = torch.from_numpy(pca.transform(difference_vector)[0])
        bs.append(b)

    sigma_K = torch.std(torch.stack(bs), dim=0)

    for idx, (image, _, meta) in enumerate(validation_loader):

        # Convert landmarks to standardized space
        target_points = torch.mean(meta['landmarks_per_annotator'], dim=1)[0]

        # Apply change
        predicted_points = torch.clone(target_points)
        # predicted_points[0] += torch.Tensor([0, 100])
        # predicted_points += torch.rand(target_points.size()) * 20 - 10

        standardized_predicted_points, mean, norm = standardize(predicted_points)

        # Convert that into affine space
        T = getT(X_, standardized_predicted_points)
        invT = torch.linalg.inv(T)
        affine_predicted_points = torch.matmul(standardized_predicted_points, T)

        # affine_predicted_points[0] += torch.Tensor([0, 0.2])

        # apply shape model
        difference_from_the_mean = torch.reshape(affine_predicted_points - X_, (1, -1))
        print(difference_from_the_mean)
        keys = torch.from_numpy(pca.transform(difference_from_the_mean)[0])
        components = torch.from_numpy(pca.components_)
        print(torch.matmul(keys, components))

        # bound keys
        '''
        # adjust b
        for i, e_k in enumerate(keys):
            sig_k = sigma_K[i]
            if e_k < -2 * sig_k:
                keys[i] = -2 * sig_k
            elif 2 * sig_k < e_k:
                keys[i] = 2 * sig_k
        '''

        affine_processed_points = X_ + torch.reshape(torch.matmul(keys, components), (-1, 2))

        # Convert points back into standardized space
        standardized_processed_points = torch.matmul(affine_processed_points, invT)

        # Convert back to normal space
        processed_points = (standardized_processed_points * norm) + mean

        plt.imshow(image[0, 0], cmap='gray')
        plt.scatter(target_points[:, 0], target_points[:, 1], s=5)
        plt.scatter(predicted_points[:, 0], predicted_points[:, 1], s=5)
        plt.scatter(processed_points[:, 0], processed_points[:, 1], s=5)
        plt.show()


if __name__ == '__main__':
    main()
