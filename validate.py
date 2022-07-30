import torch
import special_visualisations
import special_measurements

from utils import get_validation_message
from utils import compare_angles
from model import two_d_softmax
from evaluate import cal_radial_errors
from evaluate import use_aggregate_methods
from evaluate import get_sdr_statistics
from evaluate import combined_test_results
from evaluate import get_predicted_and_target_points
from general_visualisations import ground_truth_and_predictions


def validate(cfg, ensemble, validation_set_paths, loaders, visuals, special_visuals, measurements,
             logger, gpu_mode, print_progress=False, image_save_path=None):
    """

    :param cfg: the config file
    :param ensemble: a list of models
    :param loaders: a list of loaders for the images and ground truth
    :param validation_set_paths: a list of directories which the loaders apply to
    :param visuals: the codes for the generic visuals we would like saved
    :param special_visuals: the codes for the special visuals for this dataset we would like saved
    :param logger: where to print
    :param gpu_mode: where to run in gpu mode or not
    :param print_progress: print the X/N line
    :param image_save_path: where to save the images
    :return:
    """

    aggregated_mres_per_test_set = []
    sdr_statistics_per_test_set = []
    no_per_test_set = []

    # This loops over the validation sets
    for loader, validation_images_path in zip(loaders, validation_set_paths):

        logger.info("\n-----------Validating over {}-----------".format(validation_images_path))

        validate_over_set(ensemble, loader, visuals, special_visuals, measurements, cfg.VALIDATION,
                          gpu_mode, print_progress=print_progress, logger=logger)

        no_per_test_set.append(len(loader))

        aggregated_point_mres = [cal_radial_errors(predicted_points, target_points, mean=True) for
                                 predicted_points in aggregated_point_dict.values()]
        aggregated_mres_per_test_set.append(aggregated_point_mres)

        chosen_radial_errors = cal_radial_errors(aggregated_point_dict[cfg.VALIDATION.SDR_AGGREGATION_METHOD],
                                                 target_points)
        sdr_statistics = get_sdr_statistics(chosen_radial_errors, cfg.VALIDATION.SDR_THRESHOLDS)
        sdr_statistics_per_test_set.append(sdr_statistics)

        logger.info('\n-----------Statistics for {}-----------'.format(validation_images_path))
        msg = get_validation_message(aggregated_point_mres, cfg.TRAIN.ENSEMBLE_MODELS,
                                     cfg.VALIDATION.AGGREGATION_METHODS,
                                     cfg.VALIDATION.SDR_AGGREGATION_METHOD, cfg.VALIDATION.SDR_THRESHOLDS,
                                     sdr_statistics)

        logger.info('\n-----------Statistics for the Paper-----------')
        mean_error_per_landmark = torch.mean(chosen_radial_errors, dim=0)
        median_error_per_landmark = torch.median(chosen_radial_errors, dim=0)[0]
        max_error_per_landmark = torch.max(chosen_radial_errors, dim=0)[0]
        std_error_per_landmark = torch.std(chosen_radial_errors, dim=0)

        for j in range(mean_error_per_landmark.size()[0]):
            logger.info("Landmark {}: Mean: {:.3f}, Median: {:.3f}, Max: {:.3f}, Std: {:.3f}"
                        .format(j + 1, mean_error_per_landmark[j].item(), median_error_per_landmark[j].item(),
                                max_error_per_landmark[j].item(), std_error_per_landmark[j].item()))

        alpha_angle_differences, beta_angle_differences, alpha_icc, beta_icc, diagnostic_accuracy \
            = compare_angles(aggregated_point_dict[cfg.VALIDATION.SDR_AGGREGATION_METHOD], target_points)

        mean_aa_error = torch.mean(alpha_angle_differences, dim=0)
        median_aa_error = torch.median(alpha_angle_differences, dim=0)[0]
        min_aa_error = torch.min(alpha_angle_differences, dim=0)[0]
        max_aa_error = torch.max(alpha_angle_differences, dim=0)[0]
        std_aa_error = torch.std(alpha_angle_differences, dim=0)

        logger.info(
            "Alpha Angle Differences, Mean: {:.3f}, Median: {:.3f}, Min: {:.3f}, Max: {:.3f}, Std: {:.3f}, ICC: {:.3f}".
            format(mean_aa_error, median_aa_error, min_aa_error, max_aa_error, std_aa_error, alpha_icc))

        mean_ba_error = torch.mean(beta_angle_differences, dim=0)
        median_ba_error = torch.median(beta_angle_differences, dim=0)[0]
        min_ba_error = torch.min(beta_angle_differences, dim=0)[0]
        max_ba_error = torch.max(beta_angle_differences, dim=0)[0]
        std_ba_error = torch.std(beta_angle_differences, dim=0)

        logger.info(
            "Beta Angle Differences, Mean: {:.3f}, Median: {:.3f}, Min: {:.3f}, Max: {:.3f}, Std: {:.3f}, ICC: {:.3f}".
            format(mean_ba_error, median_ba_error, min_ba_error, max_ba_error, std_ba_error, beta_icc))

        logger.info("Diagnostic Accuracy: {:.3f}%".format(diagnostic_accuracy))

        logger.info("")
        logger.info(msg)

    if len(validation_set_paths) > 1:
        combined_aggregated_mres, combined_sdr_statistics = combined_test_results(aggregated_mres_per_test_set,
                                                                                  sdr_statistics_per_test_set,
                                                                                  no_per_test_set)

        logger.info('\n-----------Combined Statistics-----------')
        msg = get_validation_message(combined_aggregated_mres, cfg.TRAIN.ENSEMBLE_MODELS,
                                     cfg.VALIDATION.AGGREGATION_METHODS,
                                     cfg.VALIDATION.SDR_AGGREGATION_METHOD, cfg.VALIDATION.SDR_THRESHOLDS,
                                     combined_sdr_statistics)
        logger.info(msg)


def validate_over_set(ensemble, loader, visuals, special_visuals, measurements, cfg_validation, gpu_mode, print_progress=False, logger=None):

    '''
    if in gpu_mode the outer loop is the model because it
    takes time to transfer models to and from the gpu
    '''
    if gpu_mode:

        predicted_points_per_model = []
        eres_per_model = []
        dataset_target_points = []

        # In gpu mode we run through all images in each model first
        for model in ensemble:

            model = model.cuda()
            model.eval()

            model_predicted_points = []
            dataset_target_points = []
            model_eres = []

            for idx, (image, _, meta) in enumerate(loader):
                image = image.cuda()
                meta['landmarks_per_annotator'] = meta['landmarks_per_annotator'].cuda()
                meta['pixel_size'] = meta['pixel_size'].cuda()

                output = model(image.float())
                output = two_d_softmax(output)

                predicted_points, target_points, eres \
                    = get_predicted_and_target_points(output, meta['landmarks_per_annotator'], meta['pixel_size'])
                model_predicted_points.append(predicted_points)
                dataset_target_points.append(target_points)
                model_eres.append(eres)

                if print_progress:
                    if (idx + 1) % 1 == 0:
                        logger.info("[{}/{}]".format(idx + 1, len(loader)))

            # move model back to cpu
            model.cpu()

            model_predicted_points = torch.cat(model_predicted_points)
            dataset_target_points = torch.cat(dataset_target_points)
            model_eres = torch.cat(model_eres)
            # D = Dataset size
            # predicted_points has size [D, N, 2]
            # eres has size [D, N]
            # target_points has size [D, N, 2]

            predicted_points_per_model.append(model_predicted_points)
            eres_per_model.append(model_eres)

        # predicted_points_per_model is size [M, D, N, 2]
        # eres_per_model is size [M, D, N]
        # target_points is size [D, N, 2]
        predicted_points_per_model = torch.stack(predicted_points_per_model)
        eres_per_model = torch.stack(eres_per_model)

        aggregated_point_dict = use_aggregate_methods(predicted_points_per_model, eres_per_model,
                                                      aggregate_methods=cfg_validation.AGGREGATION_METHODS)
        aggregated_points = aggregated_point_dict[cfg_validation.SDR_AGGREGATION_METHOD]

        radial_errors = cal_radial_errors(aggregated_points, dataset_target_points)

        # quick pass through the images
        for idx, (image, _, meta) in enumerate(loader):

            radial_errors_idx = radial_errors[idx]
            target_points_idx = dataset_target_points[idx]
            aggregated_points_idx = aggregated_points[idx]

            b = 0

            name = meta['file_name'][b]
            txt = "[{}/{}] {}:\t".format(idx + 1, len(loader), name)
            for err in radial_errors_idx:
                txt += "{:.2f}\t".format(err.item())
            txt += "Avg: {:.2f}\t".format(radial_errors_idx.item())

            for measurement in measurements:
                func = eval("special_measurements." + measurement)
                predicted_angle = func(aggregated_points_idx)
                target_angle = func(target_points_idx)
                dif = abs(target_angle - predicted_angle)
                txt += "{}: {:.2f}\t".format(measurement, dif)

            logger.info(txt)

            # display predictions and ground truth
            if "pg" in visuals:
                ground_truth_and_predictions(image[b].detach().numpy(),
                                             aggregated_points_idx.detach().numpy(),
                                             target_points_idx.detach().numpy())

            for visual_name in special_visuals:
                eval("special_visualisations." + visual_name)(image[b].detach().numpy(),
                                                              aggregated_points_idx.detach().numpy(),
                                                              target_points_idx.detach().numpy())

    else:

        for idx, (image, _, meta) in enumerate(loader):

            image_predicted_points = []
            image_target_points = []
            image_eres = []

            for model in ensemble:

                model.eval()
                output = model(image.float())
                output = two_d_softmax(output)
                predicted_points, target_points, eres \
                    = get_predicted_and_target_points(output, meta['landmarks_per_annotator'], meta['pixel_size'])
                image_predicted_points.append(predicted_points)
                image_target_points.append(target_points)
                image_eres.append(eres)

            # put these arrays into a format suitable for the aggregate methods function
            image_predicted_points = torch.unsqueeze(torch.cat(image_predicted_points), 1).float()
            image_target_points = torch.unsqueeze(torch.cat(image_target_points), 1).float()
            image_eres = torch.unsqueeze(torch.cat(image_eres), 1).float()

            aggregated_point_dict = use_aggregate_methods(image_predicted_points, image_eres,
                                                          aggregate_methods=cfg_validation.AGGREGATION_METHODS)

            aggregated_points = aggregated_point_dict[cfg_validation.SDR_AGGREGATION_METHOD]

            # assumes the batch size is 1
            b = 0

            radial_errors = cal_radial_errors(aggregated_points, target_points)[b]
            avg_radial_error = torch.mean(radial_errors[b])

            name = meta['file_name'][b]
            txt = "[{}/{}] {}:\t".format(idx + 1, len(loader), name)
            for err in radial_errors:
                txt += "{:.2f}\t".format(err.item())
            txt += "Avg: {:.2f}\t".format(avg_radial_error.item())

            for measurement in measurements:
                func = eval("special_measurements." + measurement)
                predicted_angle = func(aggregated_points[b])
                target_angle = func(target_points[b])
                dif = abs(target_angle - predicted_angle)
                txt += "{}: {:.2f}\t".format(measurement, dif)

            logger.info(txt)

            # TODO: If singular experiment print out heatmaps and eres

            # display predictions and ground truth
            if "pg" in visuals:
                ground_truth_and_predictions(image[b].detach().numpy(),
                                             aggregated_points[b].detach().numpy(),
                                             target_points[b].detach().numpy())

            for visual_name in special_visuals:
                eval("special_visualisations." + visual_name)(image[b].detach().numpy(),
                                                       aggregated_points[b].detach().numpy(),
                                                       target_points[b].detach().numpy())
