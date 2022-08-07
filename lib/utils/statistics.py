import numpy as np


def get_icc(predictions, targets):
    # formula for ICC
    N = len(predictions)
    x_ = (np.sum(predictions) + np.sum(targets)) / (2.0 * N)
    s_2 = (np.sum(np.power(predictions - x_, 2)) + np.sum(np.power(targets - x_, 2))) / (2.0 * N)
    icc = np.sum((predictions - x_) * (targets - x_)) / (N * s_2)
    return icc


def get_stats(predictions, targets):
    predictions = predictions.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()

    differences = np.abs(predictions - targets)
    avg = np.mean(differences)
    std = np.std(differences)
    icc = get_icc(predictions, targets)

    return avg, std, icc