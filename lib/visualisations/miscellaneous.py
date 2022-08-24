import numpy as np
import matplotlib.pyplot as plt


def display_measurement_distribution(values, measurement_name, color, save_path):
    title = measurement_name.replace('_', ' ').capitalize()
    plt.title(title)
    plt.ylabel("Frequency")
    plt.xlabel("Measurement Value")
    plt.hist(values, bins=10, color=color)
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def display_ks_score_of_partition(partition, value_per_id):
    """

    :param partition: A dictionary of 3 keys each associated with a list
    :param value_per_id: a dictionary of ids to values
    :return:
    """

    keys = list(value_per_id.keys())
    values = list(value_per_id.values())

    xs = np.linspace(np.min(values) - 5, np.max(values) + 5, num=1000)

    stack = np.stack(best_ys_for_each_label)
    ptp = np.ptp(stack, axis=0)

    ks_score = np.max(ptp)
    ks_index = np.argmax(ptp)
    aa_index = xs[ks_index]

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