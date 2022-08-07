import matplotlib.pyplot as plt
import visualisations


def figure(image, predicted_points, target_points, suffix, figure_name, save=False, save_path=""):
    fig, ax = plt.subplots(1, 1)

    ax.imshow(image[0], cmap='gray')

    function_name = ".".join(["visualisations", suffix, figure_name])
    eval(function_name)(ax, predicted_points, target_points)

    ax.axis('off')

    if save:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def gt_and_preds(ax, predicted_points, target_points):

    ax.scatter(predicted_points[:, 0], predicted_points[:, 1], color='red', s=5)
    ax.scatter(target_points[:, 0], target_points[:, 1], color='green', s=5)

    for i, positions in enumerate(target_points):
        ax.text(positions[0], positions[1], "{}".format(i + 1), color="yellow", fontsize="small")


