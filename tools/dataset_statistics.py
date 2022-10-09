import argparse
import glob

import numpy as np

from skimage import io
from tqdm import tqdm

from lib.utils import prepare_for_dataset_preperation


def parse_args():
    parser = argparse.ArgumentParser(description='Train a network to detect landmarks')

    parser.add_argument('--cfg',
                        help='The path to the configuration file for the experiment',
                        required=True,
                        type=str)

    parser.add_argument('--images',
                        help='The path to the images',
                        type=str,
                        required=True,
                        default='')

    parser.add_argument('--annotations',
                        help='The path to the directory where annotations are stored',
                        type=str,
                        required=True,
                        default='')

    args = parser.parse_args()

    return args


def main():

    # get arguments
    args = parse_args()

    cfg, output_path = prepare_for_dataset_preperation(args.cfg)

    image_paths = sorted(glob.glob(args.images + "/*"))
    print("No. of Images: {}".format(len(image_paths)))

    widths = []
    heights = []
    avg_intensities = []

    for image_path in tqdm(image_paths):
        image = io.imread(image_path, as_gray=True)
        h, w = image.shape

        widths.append(w)
        heights.append(h)

        '''
        avg_intensity = np.mean(image)
        if avg_intensity > 1:
            avg_intensity /= 255
        avg_intensities.append(avg_intensity)
        '''

    mean_width = np.mean(widths)
    std_width = np.std(widths)

    mean_height = np.mean(heights)
    std_height = np.std(heights)

    #mean_avg_intensity = np.mean(avg_intensities)
    #std_avg_intensity = np.std(avg_intensities)

    print("Resolution range: width:{:.0f}\u00B1{:.0f} - height:{:.0f}\u00B1{:.0f}".format(mean_width, std_width,
                                                                          mean_height, std_height))

    #print("Average intensity: {:.3f}\u00B1{:.3f}".format(mean_avg_intensity, std_avg_intensity))

    annotations_paths = sorted(glob.glob(args.annotations + "/*.txt"))
    cfg_dataset = cfg.DATASET

    all_positions = []

    percentage_areas = []
    centers_x = []
    centers_y = []

    for j, annotations_path in enumerate(annotations_paths):
        kps_np_array = np.loadtxt(annotations_path, usecols=cfg_dataset.USE_COLS,
                                  delimiter=cfg_dataset.DELIMITER, max_rows=cfg_dataset.KEY_POINTS)

        if cfg_dataset.FLIP_AXIS:
            kps_np_array = np.flip(kps_np_array, axis=1)

        all_positions.append(kps_np_array)
        # calculate bound box
        min_x = np.min(kps_np_array[:, 0])
        max_x = np.max(kps_np_array[:, 0])
        min_y = np.min(kps_np_array[:, 1])
        max_y = np.max(kps_np_array[:, 1])
        bounding_box_area = (max_x - min_x) * (max_y - min_y)

        area = widths[j] * heights[j]
        percentage_area = bounding_box_area / area
        percentage_areas.append(percentage_area)

        absolute_center_x = np.mean(kps_np_array[:, 1]) / widths[j]
        absolute_center_y = np.mean(kps_np_array[:, 0]) / heights[j]
        centers_x.append(absolute_center_x)
        centers_y.append(absolute_center_y)

    all_positions = np.array(all_positions)
    all_positions = np.divide(all_positions, np.array([mean_width, mean_height]))
    std_positions = np.std(all_positions, axis=0)
    average_std = np.mean(std_positions)
    print("Std of landmarks over dataset: {:.1f}%".format(average_std * 100))

    mean_percentage_area = np.mean(percentage_areas)

    print("Percentage Area: {:.1f}%".format(mean_percentage_area * 100))

    std_x = np.std(centers_x)
    std_y = np.std(centers_y)

    print("Std x: {:.1f}% Std y: {:.1f}%".format(std_x * 100, std_y * 100))




if __name__ == '__main__':
    main()