import os
import torch

from lib.utils import get_stats
from lib.models import two_d_softmax
from lib.core.evaluate import cal_radial_errors
from lib.core.evaluate import use_aggregate_methods
from lib.core.evaluate import get_predicted_and_target_points

from lib.visualisations import intermediate_figure
from lib.visualisations import final_figure
from lib.measures import measure


def validate_over_set(ensemble, loader, visuals, cfg_validation, save_path,
                      logger=None, training_mode=False):

    predicted_points_per_model = []
    eres_per_model = []
    dataset_target_points = []
    scaled_predicted_points_per_model = []
    dataset_target_scaled_points = []

    # Create folders for images
    for visual_name in visuals:
        visual_dir = os.path.join(save_path, visual_name)
        if not os.path.exists(visual_dir):
            os.makedirs(visual_dir)

    # In gpu mode we run through all images in each model first
    for model_idx, model in enumerate(ensemble):

        model = model.cuda()
        model.eval()

        model_predicted_points = []
        model_predicted_scaled_points = []
        dataset_target_points = []
        dataset_target_scaled_points = []
        model_eres = []

        for idx, (image, _, meta) in enumerate(loader):

            # allocate
            image = image.cuda()
            meta['landmarks_per_annotator'] = meta['landmarks_per_annotator'].cuda()
            meta['pixel_size'] = meta['pixel_size'].cuda()

            output = model(image.float())
            output = two_d_softmax(output)

            predicted_points, target_points, eres, scaled_predicted_points, scaled_target_points \
                = get_predicted_and_target_points(output, meta['landmarks_per_annotator'], meta['pixel_size'])
            model_predicted_points.append(predicted_points)
            dataset_target_points.append(target_points)
            model_predicted_scaled_points.append(scaled_predicted_points)
            dataset_target_scaled_points.append(scaled_target_points)
            model_eres.append(eres)

            # print intermediate figures
            b = 0
            for visual_name in visuals:
                image_name = meta["file_name"][b]
                figure_save_path = os.path.join(save_path, visual_name,
                                                "{}_{}_{}".format(image_name, model_idx, visual_name))
                intermediate_figure(image[b], output[b].detach().cpu().numpy(),
                                    predicted_points[b].detach().cpu().numpy(),
                                    target_points[b].detach().cpu().numpy(), eres[b].detach().cpu().numpy(),
                                    visual_name, save=True, save_path=figure_save_path)

            if (idx + 1) % 30 == 0:
                logger.info("[{}/{}]".format(idx + 1, len(loader)))

        # move model back to cpu
        model.cpu()

        model_predicted_points = torch.cat(model_predicted_points)
        dataset_target_points = torch.cat(dataset_target_points)
        model_predicted_scaled_points = torch.cat(model_predicted_scaled_points)
        dataset_target_scaled_points = torch.cat(dataset_target_scaled_points)
        model_eres = torch.cat(model_eres)
        # D = Dataset size
        # predicted_points has size [D, N, 2]
        # eres has size [D, N]
        # target_points has size [D, N, 2]

        predicted_points_per_model.append(model_predicted_points)
        scaled_predicted_points_per_model.append(model_predicted_scaled_points)
        eres_per_model.append(model_eres)

    # predicted_points_per_model is size [M, D, N, 2]
    # eres_per_model is size [M, D, N]
    # target_points is size [D, N, 2]
    predicted_points_per_model = torch.stack(predicted_points_per_model).float()
    scaled_predicted_points_per_model = torch.stack(scaled_predicted_points_per_model).float()
    dataset_target_points = dataset_target_points.float()
    dataset_target_scaled_points = dataset_target_scaled_points.float()
    eres_per_model = torch.stack(eres_per_model).float()

    aggregated_point_dict = use_aggregate_methods(predicted_points_per_model, eres_per_model,
                                                  aggregate_methods=cfg_validation.AGGREGATION_METHODS)
    aggregated_points = aggregated_point_dict[cfg_validation.SDR_AGGREGATION_METHOD]
    aggregated_sacled_point_dict = use_aggregate_methods(scaled_predicted_points_per_model, eres_per_model,
                                                  aggregate_methods=cfg_validation.AGGREGATION_METHODS)
    aggregated_scaled_points = aggregated_sacled_point_dict[cfg_validation.SDR_AGGREGATION_METHOD]

    radial_errors = cal_radial_errors(aggregated_scaled_points, dataset_target_scaled_points)

    measurements_dict = {}
    for measurement in cfg_validation.MEASUREMENTS:
        measurements_dict[measurement] = []

    # quick pass through the images at the end
    for idx, (image, _, meta) in enumerate(loader):

        radial_errors_idx = radial_errors[idx]
        target_points_idx = dataset_target_points[idx]
        aggregated_points_idx = aggregated_points[idx]

        b = 0

        name = meta['file_name'][b]
        txt = "[{}/{}] {}:\t".format(idx + 1, len(loader), name)
        for err in radial_errors_idx:
            txt += "{:.2f}\t".format(err.item())
        txt += "Avg: {:.2f}\t".format(torch.mean(radial_errors_idx).item())

        for measurement in cfg_validation.MEASUREMENTS:
            predicted_angle, target_angle, dif = measure(aggregated_points_idx, target_points_idx,
                                cfg_validation.MEASUREMENTS_SUFFIX, measurement)
            measurements_dict[measurement].append([predicted_angle, target_angle])
            txt += "{}: {:.2f}\t".format(measurement, dif)

        if not training_mode:
            logger.info(txt)

        # display visuals
        for visual_name in visuals:
            image_name = meta["file_name"][b]
            figure_save_path = os.path.join(save_path, visual_name,
                                            "{}_{}".format(image_name, visual_name))
            final_figure(image[b], aggregated_points_idx.detach().cpu().numpy(),
                         aggregated_point_dict, target_points_idx.detach().cpu().numpy(),
                         cfg_validation.MEASUREMENTS_SUFFIX, visual_name,
                         save=True, save_path=figure_save_path)

    # Overall Statistics
    logger.info("\n-----------Final Statistics-----------")
    # Print average landmark localisations
    txt = "Landmark Localisations:\t"
    avg_per_landmark = torch.mean(radial_errors, dim=0)
    for avg_for_landmark in avg_per_landmark:
        txt += "{:.2f}\t".format(avg_for_landmark.item())
    overall_avg = torch.mean(radial_errors)
    txt += "Avg: {:.2f}\t".format(overall_avg.item())

    for measurement in cfg_validation.MEASUREMENTS:
        measurements_dict[measurement] = torch.Tensor(measurements_dict[measurement])
        avg, std, icc = get_stats(measurements_dict[measurement][:, 0], measurements_dict[measurement][:, 1])
        txt += "{}: [{:.2f}, {:.2f}, {:.2f}]\t".format(measurement, avg, std, icc)

    logger.info(txt)
