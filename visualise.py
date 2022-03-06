import numpy as np
import matplotlib.pyplot as plt


def visualise_ensemble(images, output_per_model, target_points, eres_per_model, keypoint_idx):

    s = 0

    # Display image
    image = images[s, 0]

    plt.imshow(image, cmap='gray', vmin=0.0, vmax=255.0)
    plt.show()

    target_points = target_points[s]
    output_per_modes = np.array(output_per_model).squeeze()
    eres_per_model = np.array(eres_per_model).squeeze()

    models, keypoints, h, w = output_per_modes.shape
    #for keypoint_idx in range(keypoints):
    # show the heatmaps per model

    relevant_heatmaps = output_per_modes[:, keypoint_idx]
    eres_for_this_landmark = eres_per_model[:, keypoint_idx]

    # show individual heatmaps
    predicted_points = []
    for model_idx, (model_heatmap, ere) in enumerate(zip(relevant_heatmaps, eres_for_this_landmark)):
        print(model_idx, ere)
        plt.imshow(image, cmap='gray', vmin=0.0, vmax=255.0)
        plt.imshow(model_heatmap, cmap='inferno', alpha=0.4)
        predicted_point = get_hottest_point(model_heatmap)
        predicted_points.append(predicted_point)
        # plt.text(predicted_point[0] + 5, predicted_point[1] + 5, "ERE: {:.2f}".format(ere), color='darkorange')
        plt.scatter([predicted_point[0]], predicted_point[1], color='red', s=100)
        # plt.text(predicted_point[0] - 5, predicted_point[1] - 5, "P{}".format(model_idx + 1), color='red')
        # plt.arrow(predicted_point[0] - 5, predicted_point[1] - 5, 5, 5, shape='right', color='red')
        plt.show()

    # show all predictions
    predicted_points = np.array(predicted_points)
    plt.imshow(image * 0.6, cmap='gray', vmin=0.0, vmax=255.0)
    plt.scatter(predicted_points[:, 0], predicted_points[:, 1], color='red', s=100)
    mean_point = np.mean(predicted_points, axis=0)
    plt.scatter(mean_point[0], mean_point[1], color='violet', s=100)
    rec_weighted_point = get_recp_weighted_point(eres_for_this_landmark, predicted_points)
    plt.scatter(rec_weighted_point[0], rec_weighted_point[1], color='cyan', s=100)
    plt.scatter(target_points[keypoint_idx, 0], target_points[keypoint_idx, 1], color='lime', s=100)
    plt.show()

    '''
    predicted_points = []
    for model_heatmap, color in zip(relevant_heatmaps, ['lightcoral', 'red', 'tomato']):
        plt.imshow(model_heatmap, cmap='inferno', alpha=0.4)
        plt.contour(model_heatmap / np.max(model_heatmap), levels=[0.2], colors=[color], zorder=2)
        predicted_point = get_hottest_point(model_heatmap)
        predicted_points.append(predicted_point)
        plt.scatter([predicted_point[0]], predicted_point[1], color=color, marker='x', zorder=2)
    #plt.imshow(squashed_heatmap, cmap='inferno', alpha=0.4)
    predicted_points = np.array(predicted_points)
    mean_point = np.mean(predicted_points, axis=0)
    eres_for_this_landmark = eres_per_model[:, keypoint_idx]
    sum_weighted_point = get_sum_weighted_point(eres_for_this_landmark, predicted_points)
    max_weighted_point = get_max_weighted_point(eres_for_this_landmark, predicted_points)
    rec_weighted_point = get_recp_weighted_point(eres_for_this_landmark, predicted_points)

    #xs = [target_points[keypoint_idx, 0], mean_point[0], sum_weighted_point[0], max_weighted_point[0], rec_weighted_point[0]]
    #ys = [target_points[keypoint_idx, 1], mean_point[1], sum_weighted_point[1], max_weighted_point[1], rec_weighted_point[1]]
    xs = [target_points[keypoint_idx, 0], mean_point[0], rec_weighted_point[0]]
    ys = [target_points[keypoint_idx, 1], mean_point[1], rec_weighted_point[1]]
    plt.scatter(xs, ys, color=['g', 'g', 'r'], zorder=2)
    plt.plot([target_points[keypoint_idx, 0], rec_weighted_point[0]],
             [target_points[keypoint_idx, 1], rec_weighted_point[1]], color='yellow')
    '''
    # dist = np.linalg.norm(target_points[keypoint_idx] - rec_weighted_point)
    # plt.text(target_points[keypoint_idx, 0], target_points[keypoint_idx, 1], "{:.2f}".format(dist), color="yellow", fontsize="small")

    # plt.show()


def visualise_ensemble_2(images, output_per_model, recp_points, target_points, eres_per_model):

    recp_points = recp_points[0]
    recp_points = np.divide(recp_points, np.array([0.2995426113372042, 0.2995426113372042]))

    s = 0

    # Display image
    image = images[s, 0]

    plt.imshow(image, cmap='gray', vmin=0.0, vmax=255.0)

    target_points = target_points[s]
    output_per_modes = np.array(output_per_model).squeeze()

    relevant_heatmaps = output_per_modes[0]
    squashed_heatmaps = np.max(relevant_heatmaps, axis=0)
    plt.imshow(squashed_heatmaps, cmap='inferno', alpha=0.4)

    for i, recp_point in enumerate(recp_points):
        j = i + 1
        offset = (5, 5)
        if j == 11:
            offset = (-30, 5)
        if j == 8:
            offset = (0, 20)
        if j == 9:
            offset = (5, 10)
        plt.text(recp_point[0] + offset[0], recp_point[1] + offset[1], "{}".format(j), color="yellow", fontsize="small")

    plt.scatter(target_points[:, 0], target_points[:, 1], color='lime', s=10)
    plt.scatter(recp_points[:, 0], recp_points[:, 1], color='red', s=10)

    plt.show()