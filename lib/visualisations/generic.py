import lib
import numpy as np
import matplotlib.pyplot as plt


def figure(image, graphics_function, args, save=False, save_path=""):
    fig, ax = plt.subplots(1, 1)

    ax.imshow(image[0], cmap='gray')

    graphics_function(ax, *args)

    ax.axis('off')

    if save:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def preliminary_figure(image, channels, target_points, figure_name, save=False, save_path=""):

    if figure_name == "show_channels":
        figure(image, show_channels, (channels, target_points))


def intermediate_figure(image, output, predicted_points, target_points, eres,
                        figure_name, save=False, save_path=""):

    if figure_name == "heatmaps_and_ere":
        figure(image, heatmaps_and_ere, (output, predicted_points, target_points, eres))


def final_figure(image, aggregated_points, aggregated_point_dict, target_points, suffix,
                 figure_name, save=False, save_path=""):

    # search for generic figure names
    if figure_name == "gt_and_preds":
        figure(image, gt_and_preds, (aggregated_points, target_points))
    elif figure_name == "aggregates":
        figure(image, aggregates, (aggregated_point_dict, target_points))
    elif figure_name == "heatmaps_and_ere":
        return
    else:
        graphics_function = eval(".".join(["lib", "visualisations", suffix, figure_name]))
        figure(image, graphics_function, (aggregated_points, target_points))


def gt_and_preds(ax, predicted_points, target_points, show_indices=True):

    ax.scatter(predicted_points[:, 0], predicted_points[:, 1], color='red', s=5)
    ax.scatter(target_points[:, 0], target_points[:, 1], color='green', s=5)

    if show_indices:
        for i, positions in enumerate(target_points):
            ax.text(positions[0], positions[1], "{}".format(i + 1), color="yellow", fontsize="small")


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
        ax.text(x + 3, y + 3, "{:.2f}".format(ere), color="white", fontsize=7)


def show_channels(ax, channels, target_points):

    squashed_channels = np.max(channels, axis=0)
    ax.imshow(squashed_channels, cmap='inferno', alpha=0.5)

    for i, positions in enumerate(target_points):
        ax.text(positions[0], positions[1], "{}".format(i + 1), color="yellow", fontsize="small")


