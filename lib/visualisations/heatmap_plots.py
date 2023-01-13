import math
import wandb
import numpy as np
import matplotlib.pyplot as plt

from scipy import stats
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

font_size = 25
large_font_size = 20
x_pad = 0.2
y_pad = 0.2


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
    plt.rcParams["figure.figsize"] = (6.4, 6.4)
    fig, ax = plt.subplots(1, 1)
    plt.subplots_adjust(bottom=y_pad)
    plt.subplots_adjust(left=x_pad)
    ax.grid(zorder=0)
    plt.xlabel(x_label, fontsize=font_size)
    plt.ylabel(y_label, fontsize=font_size)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.text(0.3, 0.075, "CORRELATION={:.2f}".format(correlation), backgroundcolor=(0.8, 0.8, 0.8, 0.8), size=large_font_size, transform=ax.transAxes)
    ax.scatter(binned_x_values, binned_y_values, c='lime', edgecolors='black', zorder=3, s=60)

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
    plt.rcParams["figure.figsize"] = (6.4, 6.4)
    fig, ax = plt.subplots(1, 1)
    plt.subplots_adjust(bottom=y_pad)
    plt.subplots_adjust(left=x_pad)
    ax.grid(zorder=0)
    plt.xlabel("False Positive Rate (FPR)", fontsize=font_size)
    plt.ylabel("True Positive Rate (TPR)", fontsize=font_size)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.plot([0, 1], [0, 1], c='black', linestyle='dashed')
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.grid(True)

    plt.plot(fpr, tpr, c="blue")
    plt.text(0.2, 0.075, 'Area Under Curve={:.2f}'.format(auc), backgroundcolor=(0.8, 0.8, 0.8, 0.8), size=large_font_size,
             transform=ax.transAxes)

    wb_image = wandb.Image(plt)
    plt.close()

    return proposed_threshold, auc, wb_image


# At the moment this assumes all images have the same resolution
def reliability_diagram(radial_errors, mode_probabilities, pixel_size, n_of_bins=10):

    x_min = np.quantile(mode_probabilities, 0.10)
    x_max = np.quantile(mode_probabilities, 0.90)
    bins = np.linspace(x_min, x_max, n_of_bins + 1)
    bins[0] = 0.0
    bins[-1] = 1.1
    widths = (x_max - x_min) / n_of_bins
    radius = np.sqrt(np.square(pixel_size) / math.pi)
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
    avg_conf_for_each_bin = np.divide(total_confidence_for_each_bin, count_for_each_bin.astype(float),
                                      out=np.zeros_like(total_confidence_for_each_bin), where=count_for_each_bin != 0)
    avg_acc_for_each_bin = np.divide(no_of_correct_preds, count_for_each_bin.astype(float),
                                      out=np.zeros_like(total_confidence_for_each_bin), where=count_for_each_bin != 0)

    # print(avg_acc_for_each_bin, avg_conf_for_each_bin, count_for_each_bin.astype(float))

    n = float(np.sum(count_for_each_bin))
    ece = 0.0
    for i in range(len(bins) - 1):
        ece += count_for_each_bin[i] / n * np.abs(avg_acc_for_each_bin[i] - avg_conf_for_each_bin[i])
    ece *= 100

    # save plot
    plt.rcParams["figure.figsize"] = (6.4, 4.8)
    fig, ax = plt.subplots(1, 1)
    ax.grid(zorder=0)

    plt.subplots_adjust(left=x_pad)
    plt.xlabel('Confidence', fontsize=font_size)
    plt.ylabel('Accuracy', fontsize=font_size)
    plt.xticks(fontsize=font_size)
    plt.locator_params(axis='x', nbins=5)
    plt.subplots_adjust(bottom=y_pad)
    plt.yticks(fontsize=font_size)
    plt.grid(zorder=0)
    plt.xlim(x_min, x_max)
    plt.ylim(0.0, max(np.max(avg_acc_for_each_bin), np.max(avg_conf_for_each_bin)) * 1.1)
    plt.bar(bins[:-1], avg_acc_for_each_bin, align='edge', width=widths, color='blue', edgecolor='black', label='Accuracy', zorder=3)
    plt.bar(bins[:-1], avg_conf_for_each_bin, align='edge', width=widths, color='lime', edgecolor='black', alpha=0.5,
            label='Gap', zorder=3)
    plt.legend(fontsize=20, loc="upper left", prop={'size': 20})
    plt.text(0.45, 0.075, 'ECE={:.2f}'.format(ece), backgroundcolor=(0.8, 0.8, 0.8, 0.8), fontsize=30, transform=ax.transAxes)

    wb_image = wandb.Image(plt)
    plt.close()

    return ece, wb_image
