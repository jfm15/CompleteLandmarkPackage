import math
import numpy as np
import matplotlib.pyplot as plt

from scipy import optimize


# Get the predicted landmark point from the coordinate of the hottest point
def get_hottest_point(heatmap):
    w, h = heatmap.shape
    flattened_heatmap = np.ndarray.flatten(heatmap)
    hottest_idx = np.argmax(flattened_heatmap)
    return np.flip(np.array(np.unravel_index(hottest_idx, [w, h])))


def get_mode_probability(heatmap):
    return np.max(heatmap)


def get_optimal_sigma(radial_errors):
    def f(sigma):
        sum = 0
        constant = math.log(1 / (sigma * math.sqrt(2 * math.pi)))
        for radial_error in radial_errors:
            if radial_error > 0:
                sum += constant + (-0.5 * (radial_error / sigma) ** 2)
        return -sum

    initial_guess = np.mean(radial_errors)
    optimal = optimize.minimize_scalar(f, bounds=[initial_guess/2, initial_guess*4], method="bounded")
    return optimal.x


def calculate_ere(heatmap, predicted_point_scaled, pixel_size, significant_pixel_cutoff=0.05):
    normalized_heatmap = heatmap / np.max(heatmap)
    normalized_heatmap = np.where(normalized_heatmap > significant_pixel_cutoff, normalized_heatmap, 0)
    normalized_heatmap /= np.sum(normalized_heatmap)
    indices = np.argwhere(normalized_heatmap)
    ere = 0
    for twod_idx in indices:
        scaled_idx = np.flip(twod_idx) * pixel_size
        dist = np.linalg.norm(predicted_point_scaled - scaled_idx)
        ere += dist * normalized_heatmap[twod_idx[0], twod_idx[1]]
    return ere


def get_predicted_points(heatmap_stack):

    batch_size, no_of_key_points, w, h = heatmap_stack.shape
    predicted_points = np.zeros((batch_size, no_of_key_points, 2))

    for i in range(batch_size):
        for j in range(no_of_key_points):

            # Get predicted point
            predicted_points[i, j] = get_hottest_point(heatmap_stack[i, j])

    return predicted_points


def evaluate(heatmap_stack, landmarks_per_annotator, pixels_sizes):
    batch_size, no_of_key_points, w, h = heatmap_stack.shape
    pixel_error_per_landmark = np.zeros((batch_size, no_of_key_points))
    radial_error_per_landmark = np.zeros((batch_size, no_of_key_points))
    expected_error_per_landmark = np.zeros((batch_size, no_of_key_points))
    mode_probability_per_landmark = np.zeros((batch_size, no_of_key_points))

    for i in range(batch_size):

        pixel_size_for_sample = pixels_sizes[i]

        for j in range(no_of_key_points):

            # Get predicted point
            predicted_point = get_hottest_point(heatmap_stack[i, j])
            predicted_point_scaled = predicted_point * pixel_size_for_sample

            # Average the annotators to get target point
            target_point = np.mean(landmarks_per_annotator[i, :, j], axis=0)
            target_point_scaled = target_point * pixel_size_for_sample

            pixel_error = np.linalg.norm(predicted_point - target_point)
            localisation_error = np.linalg.norm(predicted_point_scaled - target_point_scaled)

            pixel_error_per_landmark[i, j] = pixel_error
            radial_error_per_landmark[i, j] = localisation_error

            expected_error_per_landmark[i, j] = calculate_ere(heatmap_stack[i, j], predicted_point_scaled,
                                                              pixel_size_for_sample)

            mode_probability_per_landmark[i, j] = get_mode_probability(heatmap_stack[i, j])

    # removed the others from this because they take too long
    return pixel_error_per_landmark, radial_error_per_landmark, expected_error_per_landmark, mode_probability_per_landmark


# This function will visualise the output of the first image in the batch
def visualise(images, channel_stack, heatmap_stack, landmarks_per_annotator):
    s = 0

    # Display image
    image = images[s, 0]
    plt.imshow(image, cmap='gray')

    # Display heatmaps
    normalized_heatmaps = heatmap_stack[s] / np.max(heatmap_stack[s], axis=(1, 2), keepdims=True)
    squashed_heatmaps = np.max(normalized_heatmaps, axis=0)
    #  plt.imshow(squashed_heatmaps, cmap='inferno', alpha=0.4)
    plt.imshow(np.max(channel_stack[s], axis=0), cmap='inferno', alpha=0.4)

    # Display predicted points
    predicted_landmark_positions = np.array([get_hottest_point(heatmap) for heatmap in normalized_heatmaps])
    plt.scatter(predicted_landmark_positions[:, 0], predicted_landmark_positions[:, 1], color='red', s=2)

    # Display ground truth points
    ground_truth_landmark_position = np.mean(landmarks_per_annotator[s], axis=0)
    plt.scatter(ground_truth_landmark_position[:, 0], ground_truth_landmark_position[:, 1], color='green', s=2)

    plt.show()


def produce_sdr_statistics(radial_errors, thresholds):
    successful_detection_rates = []
    for threshold in thresholds:
        sdr = 100 * np.sum(radial_errors < threshold) / len(radial_errors)
        successful_detection_rates.append(sdr)
    return successful_detection_rates
