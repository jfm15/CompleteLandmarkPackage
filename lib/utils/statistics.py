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
    med = np.median(differences)
    std = np.std(differences)
    icc = get_icc(predictions, targets)

    return avg, std, med, icc


def produce_sdr_statistics(radial_errors, thresholds):
    successful_detection_rates = []
    for threshold in thresholds:
        sdr = 100 * np.sum(radial_errors < threshold) / len(radial_errors)
        successful_detection_rates.append(sdr)
    return successful_detection_rates