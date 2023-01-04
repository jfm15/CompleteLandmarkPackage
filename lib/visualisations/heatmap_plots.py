import math
import wandb
import numpy as np
import matplotlib.pyplot as plt

from scipy import stats
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


def correlation_graph(x_values, y_values, x_label, y_label, n_bin=36):

    # Bin the y values and calculate the average x values for each bin
    binned_x_values = []
    binned_y_values = []
    sorted_indices = np.argsort(y_values)
    for l in range(int(len(y_values) / n_bin)):
        binned_indices = sorted_indices[l * n_bin: (l + 1) * n_bin]
        binned_y_values.append(np.mean(np.take(y_values, binned_indices)))
        binned_x_values.append(np.mean(np.take(x_values, binned_indices)))
    correlation = np.corrcoef(binned_x_values, binned_y_values)[0, 1]

    # Plot graph
    plt.rcParams["figure.figsize"] = (6, 6)
    fig, ax = plt.subplots(1, 1)
    ax.grid(zorder=0)
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.text(0.5, 0.075, "CORRELATION={:.2f}".format(correlation), backgroundcolor=(0.8, 0.8, 0.8, 0.8), size='x-large', transform=ax.transAxes)
    ax.scatter(binned_x_values, binned_y_values, c='lime', edgecolors='black', zorder=3)

    wb_image = wandb.Image(plt)
    plt.close()

    return correlation, wb_image


def roc_outlier_graph(ground_truth, predictive_feature, outlier_threshold=2.0):
    outliers = ground_truth > outlier_threshold

    fpr, tpr, thresholds = roc_curve(outliers, predictive_feature)
    auc = roc_auc_score(outliers, predictive_feature)

    first_idx = np.min(np.argwhere(tpr > 0.5))
    proposed_threshold = thresholds[first_idx]

    # Plot graph
    plt.rcParams["figure.figsize"] = (6, 6)
    fig, ax = plt.subplots(1, 1)
    ax.grid(zorder=0)
    plt.xlabel("False Positive Rate (FPR)", fontsize=14)
    plt.ylabel("True Positive Rate (TPR)", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.plot([0, 1], [0, 1], c='black', linestyle='dashed')
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.grid(True)

    plt.plot(fpr, tpr, c="blue")
    plt.text(0.42, 0.075, 'Area Under Curve={:.2f}'.format(auc), backgroundcolor=(0.8, 0.8, 0.8, 0.8), size='x-large',
             transform=ax.transAxes)

    wb_image = wandb.Image(plt)
    plt.close()

    return proposed_threshold, auc, wb_image


# At the moment this assumes all images have the same resolution
def reliability_diagram(radial_errors, mode_probabilities, n_of_bins=10, pixel_size=0.30234375):

    x_max = math.floor(np.max(mode_probabilities) / 0.01) * 0.01
    bins = np.linspace(0, x_max, n_of_bins + 1)
    bins[-1] = 1.1
    widths = x_max / n_of_bins
    radius = math.sqrt((pixel_size**2) / math.pi)
    correct_predictions = radial_errors < radius

    # a 10 length array with values adding to 19
    count_for_each_bin, _ = np.histogram(mode_probabilities, bins=bins)

    # total confidence in each bin
    total_confidence_for_each_bin, _, bin_indices \
        = stats.binned_statistic(mode_probabilities, mode_probabilities, 'sum', bins=bins)

    no_of_correct_preds = np.zeros(len(bins) - 1)
    for bin_idx, pred_correct in zip(bin_indices, correct_predictions):
        no_of_correct_preds[bin_idx - 1] += pred_correct

    # get confidence of each bin
    avg_conf_for_each_bin = total_confidence_for_each_bin / count_for_each_bin.astype(float)
    avg_acc_for_each_bin = no_of_correct_preds / count_for_each_bin.astype(float)

    n = float(np.sum(count_for_each_bin))
    ece = 0.0
    for i in range(len(bins) - 1):
        ece += count_for_each_bin[i] / n * np.abs(avg_acc_for_each_bin[i] - avg_conf_for_each_bin[i])
    ece *= 100

    # save plot
    plt.rcParams["figure.figsize"] = (6, 6)
    fig, ax = plt.subplots(1, 1)
    ax.grid(zorder=0)

    plt.subplots_adjust(left=0.15)
    plt.xlabel('Confidence', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(zorder=0)
    plt.xlim(0.0, x_max)
    plt.ylim(0.0, np.max(avg_acc_for_each_bin))
    plt.bar(bins[:-1], avg_acc_for_each_bin, align='edge', width=widths, color='blue', edgecolor='black', label='Accuracy', zorder=3)
    plt.bar(bins[:-1], avg_conf_for_each_bin, align='edge', width=widths, color='lime', edgecolor='black', alpha=0.5,
            label='Gap', zorder=3)
    plt.legend(fontsize=20, loc="upper left", prop={'size': 16})
    plt.text(0.71, 0.075, 'ECE={:.2f}'.format(ece), backgroundcolor='white', fontsize='x-large', transform=ax.transAxes)

    wb_image = wandb.Image(plt)
    plt.close()

    return ece, wb_image
