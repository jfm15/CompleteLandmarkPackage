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
    centered_points = points - torch.mean(points, dim=0, keepdim=False)
    norm1 = torch.norm(centered_points)
    return centered_points / norm1


def unstandardize(points):



# Gets the best affine transformation which maps points onto the base
def getT(points, base):
    R, S = orthogonal_procrustes(points, base)
    T = torch.matmul(torch.from_numpy(R).float(), scale_matrix(S).float())
    return T.double()


def get_principal_components(vectors, percentage=0.99):
    pca = PCA(n_components=vectors.size()[1])
    pca.fit(vectors)

    # get the vectors which account for 99% variance
    sum = 0
    i = 0
    while sum < percentage:
        sum += pca.explained_variance_ratio_[i]
        i += 1

    return pca, i


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
        standardized = standardize(target_points)
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

    pca, K = get_principal_components(difference_vectors)

    # get sigma_K
    bs = []
    for difference_vector in difference_vectors:
        difference_vector = torch.unsqueeze(difference_vector, 0)
        b = torch.from_numpy(pca.transform(difference_vector)[0, :K])
        bs.append(b)

    sigma_K = torch.std(torch.stack(bs), dim=0)

    for idx, (image, _, meta) in enumerate(validation_loader):
        target_points = torch.mean(meta['landmarks_per_annotator'], dim=1)[0]
        # target_points += torch.rand(target_points.size()) * 20
        # target_points[0] += torch.Tensor([0, 200])
        standardized = standardize(target_points)
        T = getT(X_, standardized)
        true = torch.matmul(standardized, T)
        # error = true + torch.rand(target_points.size()) * 0.04 - 0.02
        error = torch.clone(true)
        error[0] += torch.Tensor([0, 0.5])

        difference_vector = true - X_
        difference_vector = torch.reshape(torch.flatten(difference_vector), (1, -1))

        b = torch.from_numpy(pca.transform(difference_vector)[0, :K])
        components = torch.from_numpy(pca.components_[:K])

        no_adjustment = X_ + torch.reshape(torch.matmul(b, components), (-1, 2))

        # adjust b
        for i, e_k in enumerate(b):
            sig_k = sigma_K[i]
            if e_k < -3 * sig_k:
                b[i] = -3 * sig_k
            elif 3 * sig_k < e_k:
                b[i] = 3 * sig_k

        adjustment = X_ + torch.reshape(torch.matmul(b, components), (-1, 2))

        plt.scatter(true[:, 0], true[:, 1], label="true")
        plt.scatter(error[:, 0], error[:, 1], label="predicted")
        plt.scatter(no_adjustment[:, 0], no_adjustment[:, 1], label="projected")
        # plt.scatter(adjustment[:, 0], adjustment[:, 1], label="adjusted")
        plt.legend()
        plt.show()


if __name__ == '__main__':
    main()
