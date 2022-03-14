import os
import numpy as np
import matplotlib.pyplot as plt


def visualise_heatmaps(image, output, eres, predicted_pixel_points, model_idx, name, save_image_path):

    usable_image = np.squeeze(image)

    # cycle through landmarks
    no_of_landmarks = output.shape[1]
    for landmark_idx in range(no_of_landmarks):

        image_path = os.path.join(save_image_path, "heat_{}_{}_model:{}".format(name, landmark_idx + 1, model_idx))
        figure = plt.gcf()
        figure.set_size_inches(usable_image.shape[1] / 50, usable_image.shape[0] / 50)
        plt.imshow(usable_image, cmap='gray', vmin=0.0, vmax=255.0)
        plt.imshow(output[0, landmark_idx], cmap='inferno', alpha=0.4)
        pred_point = predicted_pixel_points[0, landmark_idx]
        ere = eres[0, landmark_idx]
        # plt.scatter(pred_point[0], pred_point[1], color='white', s=3)
        # plt.text(pred_point[0] + 10, pred_point[1] + 10, "ERE: {:.3f}".format(ere), color="white", fontsize="small")
        plt.axis('off')
        plt.savefig(image_path, bbox_inches='tight', dpi=100)
        plt.close()


def visualise_aggregations(image, predicted_points_per_model, rec_weighted_model_points,
                           mean_model_points, target_points, pixel_size, name, save_image_path):

    usable_image = np.squeeze(image) * 0.6

    # cycle through landmarks
    no_of_landmarks = rec_weighted_model_points.shape[0]
    for landmark_idx in range(no_of_landmarks):

        image_path = os.path.join(save_image_path, "agg_{}_{}".format(name, landmark_idx + 1))
        figure = plt.gcf()
        figure.set_size_inches(usable_image.shape[1] / 50, usable_image.shape[0] / 50)
        plt.imshow(usable_image, cmap='gray', vmin=0.0, vmax=255.0)

        for model_idx in range(len(predicted_points_per_model)):
            individual_model_pred = predicted_points_per_model[model_idx, landmark_idx]
            scaled_individual_model_pred = np.divide(individual_model_pred, pixel_size)
            plt.scatter(scaled_individual_model_pred[0], scaled_individual_model_pred[1], color='white', s=3)

        mean_pred = mean_model_points[landmark_idx]
        scaled_mean_pred = np.divide(mean_pred, pixel_size)
        plt.scatter(scaled_mean_pred[0], scaled_mean_pred[1], color='violet', s=3)

        confidence_weighted_pred = rec_weighted_model_points[landmark_idx]
        scaled_confidence_weighted_pred = np.divide(confidence_weighted_pred, pixel_size)
        plt.scatter(scaled_confidence_weighted_pred[0], scaled_confidence_weighted_pred[1], color='red', s=3)

        ground_truth = target_points[landmark_idx]
        scaled_ground_truth = np.divide(ground_truth, pixel_size)
        plt.scatter(scaled_ground_truth[0], scaled_ground_truth[1], color='lime', s=3)

        plt.axis('off')
        plt.savefig(image_path, bbox_inches='tight', dpi=100)
        plt.close()


def save_final_predictions(loader, predicted_points, target_points, save_image_path):

    for idx, (image, _, meta) in enumerate(loader):
        name = meta['file_name'][0]
        pixel_size = meta['pixel_size'][0]

        # removes the first 2 dimensions from image
        usable_image = np.squeeze(image)

        figure = plt.gcf()
        figure.set_size_inches(usable_image.shape[1] / 50, usable_image.shape[0] / 50)

        plt.imshow(usable_image, cmap='gray', vmin=0.0, vmax=255.0)

        plt.scatter(predicted_points[idx, :, 0], predicted_points[idx, :, 1], color='red', s=3)

        save_path = os.path.join(save_image_path, "{}_predictions".format(name))
        plt.axis('off')
        plt.savefig(save_path, bbox_inches='tight', dpi=100)
        plt.close()




