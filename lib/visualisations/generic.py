import wandb

import lib
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from scipy.stats import multivariate_normal


def figure(image, graphics_function, args):
    fig, ax = plt.subplots(1, 1)

    ax.imshow(image[0], cmap='gray')

    graphics_function(ax, *args)

    ax.axis('off')
    plt.tight_layout()

    h, w = np.shape(image[0])
    fig.set_size_inches(w / 100.0, h / 100.0)
    fig.set_dpi(150)

    wb_image = wandb.Image(plt)
    plt.close()

    return wb_image


def preliminary_figure(image, channels, target_points, figure_name):

    if figure_name == "show_channels":
        return figure(image, show_channels, (channels, target_points))


def intermediate_figure(image, output, predicted_points, target_points, eres, figure_name):

    if figure_name == "heatmaps_and_ere":
        return figure(image, heatmaps_and_ere, (output, predicted_points, target_points, eres))
    elif figure_name == "heatmaps":
        return figure(image, heatmaps, (output, predicted_points, target_points, eres))
    elif figure_name == "isolated_heatmap":
        return figure(image, isolated_heatmap, (output, predicted_points, target_points, eres))
    elif figure_name == "gaussian_approximation":
        return figure(image, gaussian_approximation, (output, predicted_points, target_points, eres))
    elif figure_name == "heatmaps_and_preds":
        return figure(image, heatmaps_and_preds, (output, predicted_points, target_points, eres))


def final_figure(image, aggregated_points, aggregated_point_dict, target_points, suffix, figure_name):

    # search for generic figure names
    if figure_name == "gt_and_preds":
        return figure(image, gt_and_preds, (aggregated_points, target_points))
    elif figure_name == "gt_and_preds_small":
        return figure(image, gt_and_preds_small, (aggregated_points, target_points))
    elif figure_name == "preds":
        return figure(image, preds, (aggregated_points, target_points))
    elif figure_name == "gt":
        return figure(image, targets, (aggregated_points, target_points))
    elif figure_name == "gt_no_indices":
        return figure(image, targets, (aggregated_points, target_points, False, False, False))
    elif figure_name == "gt_bounding_box":
        return figure(image, targets, (aggregated_points, target_points, True, True, False, image[0].size()))
    elif figure_name == "aggregates":
        return figure(image, aggregates, (aggregated_point_dict, target_points))
    elif figure_name in ["heatmaps", "isolated_heatmap", "gaussian_approximation",
                         "heatmaps_and_ere", "heatmaps_and_preds"]:
        return
    else:
        graphics_function = eval(".".join(["lib", "visualisations", suffix, figure_name]))
        return figure(image, graphics_function, (aggregated_points, target_points))


def gt_and_preds(ax, predicted_points, target_points, show_indices=True):

    ax.scatter(target_points[:, 0], target_points[:, 1], color='lime', s=30)
    ax.scatter(predicted_points[:, 0], predicted_points[:, 1], color='red', s=30)

    '''
    if show_indices:
        for i, positions in enumerate(target_points):
            ax.text(positions[0], positions[1], "{}".format(i + 1), color="yellow", fontsize="small")
    '''


def gt_and_preds_small(ax, predicted_points, target_points, show_indices=True):

    ax.scatter(target_points[:, 0], target_points[:, 1], color='lime', s=15)
    ax.scatter(predicted_points[:, 0], predicted_points[:, 1], color='red', s=15)

    '''
    if show_indices:
        for i, positions in enumerate(target_points):
            ax.text(positions[0], positions[1], "{}".format(i + 1), color="yellow", fontsize="small")
    '''


def preds(ax, predicted_points, target_points, show_indices=False, s=20):

    ax.scatter(predicted_points[:, 0], predicted_points[:, 1], color='red', s=s)

    if show_indices:
        for i, positions in enumerate(predicted_points):
            ax.text(positions[0] + 5, positions[1] - 5, "{}".format(i + 1), color="yellow", fontsize="x-large")


def targets(ax, predicted_points, target_points, show_bounding_box=False, show_mean_point=False, show_indices=True, image_size=None):

    if show_indices:
        for i, positions in enumerate(target_points):
            ax.text(positions[0] + 5, positions[1] - 5, "{}".format(i + 1), color="yellow", fontsize="large")

    if show_bounding_box:
        min_x = torch.min(target_points[:, 0]).item()
        max_x = torch.max(target_points[:, 0]).item()
        min_y = torch.min(target_points[:, 1]).item()
        max_y = torch.max(target_points[:, 1]).item()
        width = max_x - min_x
        height = max_y - min_y
        rect = patches.Rectangle((min_x, min_y), width, height, linewidth=1, edgecolor='gold', facecolor='none', zorder=1)
        ax.add_patch(rect)
        bounding_box_area = width * height
        area = image_size[0] * image_size[1]
        area_percentage = bounding_box_area / area

        ax.text(0.40, 0.05, "Area Percentage = {:.0f}%".format(area_percentage * 100), backgroundcolor=(0.8, 0.8, 0.8, 0.8), size='x-large',
                 transform=ax.transAxes)

    if show_mean_point:
        mean_point = torch.mean(target_points, dim=0)
        ax.scatter([mean_point[0]], [mean_point[1]], color='red', s=100, marker="x", zorder=2)


    ax.scatter(target_points[:, 0], target_points[:, 1], color='green', s=20, zorder=2)


# assumes aggregated_point_dict contains keys -> Tensors[B, N, 2] where B=1
def aggregates(ax, aggregated_point_dict, target_points):

    # Display the predictions for the base estimators
    base_estimator = 1
    while str(base_estimator) in aggregated_point_dict:
        base_estimator_prediction = aggregated_point_dict[str(base_estimator)][0]
        ax.scatter(base_estimator_prediction[:, 0], base_estimator_prediction[:, 1],
                   color='red', s=5, label='Base Estimator {}'.format(base_estimator))
        base_estimator += 1

    # Display mean predictions
    mean_predictions = aggregated_point_dict["mean average"][0]
    plt.scatter(mean_predictions[:, 0], mean_predictions[:, 1],
                color='violet', s=3, label='Mean')

    # Display weighted predictions
    weighted_predictions = aggregated_point_dict["confidence weighted"][0]
    plt.scatter(weighted_predictions[:, 0], weighted_predictions[:, 1], color='cyan', s=3, label='Weighted')

    ax.scatter(target_points[:, 0], target_points[:, 1], color='green', s=5, label='Ground Truth')
    ax.legend()


def heatmaps_and_ere(ax, output, predicted_points, target_points, eres):

    normalized_heatmaps = output / np.max(output, axis=(1, 2), keepdims=True)
    squashed_output = np.max(normalized_heatmaps, axis=0)

    ax.imshow(squashed_output, cmap='inferno', alpha=0.4)

    gt_and_preds(ax, predicted_points, target_points, show_indices=False)

    for ere, position in zip(eres, predicted_points):
        x, y = position
        ax.text(x + 3, y + 3, "{:.3f}".format(ere), color="white", fontsize=7)


def heatmaps(ax, output, predicted_points, target_points, eres):

    normalized_heatmaps = output / np.max(output, axis=(1, 2), keepdims=True)
    squashed_output = np.max(normalized_heatmaps, axis=0)

    ax.imshow(squashed_output, cmap='inferno', alpha=0.5)


# 6 for x-ray pelvis
# 20 for hand
def isolated_heatmap(ax, output, predicted_points, target_points, eres, channel=20):
    ax.imshow(output[channel], cmap='inferno', alpha=0.5)


def gaussian_approximation(ax, output, predicted_points, target_points, eres, channel=6):
    heatmap = output[channel]

    # find mean
    expected_position = np.zeros([2])
    width, height = heatmap.shape
    for x in range(width):
        for y in range(height):
            expected_position += np.array([x, y]) * heatmap[x, y]

    # find covariance matrix
    ex_x, ex_y = expected_position
    var_x = 0
    var_y = 0
    cov_xy = 0
    for x in range(width):
        for y in range(height):
            var_x += (x - ex_x)**2 * heatmap[x, y]
            var_y += (y - ex_y) ** 2 * heatmap[x, y]
            cov_xy += (x - ex_x) * (y - ex_y) * heatmap[x, y]

    ax.imshow(heatmap, cmap='inferno', alpha=0.5)
    ax.scatter([expected_position[1]], [expected_position[0]], s=5, color="red")

    x, y = np.meshgrid(range(width), range(height))
    pos = np.dstack((x, y))
    rv = multivariate_normal(expected_position, [[var_x, cov_xy], [cov_xy, var_y]])
    pdf = rv.pdf(pos)
    ax.contour(y, x, pdf, levels=[np.max(pdf) / 3], linewidths=1, colors="yellow")


def heatmaps_and_preds(ax, output, predicted_points, target_points, eres):

    normalized_heatmaps = output / np.max(output, axis=(1, 2), keepdims=True)
    squashed_output = np.max(normalized_heatmaps, axis=0)

    ax.imshow(squashed_output, cmap='inferno', alpha=0.4)

    preds(ax, predicted_points, target_points, show_indices=False, s=2)


def show_channels(ax, channels, target_points, custom=20):
    ax.imshow(channels[custom], cmap='inferno', alpha=0.4)
    '''
    squashed_channels = np.max(channels, axis=0)
    ax.imshow(squashed_channels, cmap='inferno', alpha=0.5)

    for i, positions in enumerate(target_points):
        ax.text(positions[0], positions[1], "{}".format(i + 1), color="yellow", fontsize="small")
    '''


