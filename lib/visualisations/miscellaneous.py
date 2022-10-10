import os
import numpy as np
import matplotlib.pyplot as plt


def display_measurement_distribution(values, measurement_name, color):
    title = measurement_name.replace('_', ' ').capitalize()
    plt.title(title)
    plt.ylabel("Frequency")
    plt.xlabel("Measurement Value")
    plt.hist(values, bins=10, color=color)
    plt.grid(True)
    plt.show()


def display_ks_score_of_partition(partition, value_per_id, measurement_name):
    """

    :param partition: A dictionary of 3 keys each associated with a list
    :param value_per_id: a dictionary of ids to values
    :return:
    """

    values = list(value_per_id.values())

    partition_labels = ["training", "validation", "testing"]

    xs = np.linspace(np.min(values) - 4, np.max(values) + 4, num=1000)

    # Create a cdf
    ys_for_each_label = []
    for label in partition_labels:
        xp = []
        for id in partition[label]:
            xp.append(value_per_id[id])
        xp = np.sort(xp)
        fp = np.linspace(0, 1, num=len(xp))
        ys = np.interp(xs, xp, fp)
        ys_for_each_label.append(ys)

    stack = np.stack(ys_for_each_label)
    ptp = np.ptp(stack, axis=0)

    ks_score = np.max(ptp)
    ks_index = np.argmax(ptp)
    aa_index = xs[ks_index]

    plt.figure(
        figsize=(6, 6))
    fig, ax = plt.subplots()
    # plt.rcParams["font.weight"] = "bold"
    plt.rcParams["font.size"] = "12"
    # Convert from snake case
    measurement_name_split = measurement_name.split("_")
    measurement_name = " ".join([x.capitalize() for x in measurement_name_split])
    plt.xlabel(measurement_name, fontsize=12)
    plt.ylabel('Cumulative Proportion of Dataset', fontsize=12)
    colors = ['r', 'g', 'b']
    for i, ys in enumerate(ys_for_each_label):
        plt.plot(xs, ys, (colors[i] + "-"), label=partition_labels[i])
    x = aa_index
    y = np.min(stack[:, ks_index])
    dx = 0
    dy = np.ptp(stack[:, ks_index])
    plt.arrow(x, y, 0, dy, color='k', length_includes_head=True, head_width=0.75, head_length=0.01, linewidth=2, zorder=3)
    plt.arrow(x, y + dy, 0, -dy, color='k', length_includes_head=True, head_width=0.75, head_length=0.01, linewidth=2, zorder=3)
    plt.text(0.55, 0.05, "KS test = {:.3f}".format(ks_score), backgroundcolor=(0.8, 0.8, 0.8, 0.8), size='x-large',
             transform=ax.transAxes)
    plt.legend()
    plt.grid(True)
    plt.show()


def display_ks_scores(ks_scores):
    counts, bins = np.histogram(ks_scores, bins=50)
    counts = counts.astype(float) / len(ks_scores)
    plt.stairs(counts, bins, color='b', fill=True)
    # plt.hist(ks_scores, bins=50, density=True, color='b')
    plt.xlabel('KS Score', fontsize=12)
    plt.ylabel('Proportion of Subset Splits', fontsize=12)
    plt.grid(True)
    plt.show()


def display_box_plot(radial_errors, save_path):
    plt.boxplot(radial_errors)
    plt.savefig(save_path)
    plt.close()