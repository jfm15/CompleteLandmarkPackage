import os
import torch

from lib.utils import get_stats
from lib.core.evaluate import cal_radial_errors
from lib.core.evaluate import use_aggregate_methods
from lib.core.evaluate import get_predicted_and_target_points
from lib.core.evaluate import get_sdr_statistics

from lib.visualisations import intermediate_figure
from lib.visualisations import final_figure
from lib.visualisations import display_box_plot
from lib.visualisations import radial_error_vs_ere_graph
from lib.visualisations import reliability_diagram
from lib.visualisations import roc_outlier_graph

from lib.measures import measure
from lib.measures import diagnose_set


def validate_over_set(ensemble, loader, final_layer, loss_function, visuals, cfg_validation, save_path,
                      logger=None, training_mode=False, temperature_scaling_mode=False):

    predicted_points_per_model = []
    eres_per_model = []
    modes_per_model = []
    dataset_target_points = []
    scaled_predicted_points_per_model = []
    dataset_target_scaled_points = []
    losses = []

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
        model_modes = []

        for idx, (image, channels, meta) in enumerate(loader):

            # allocate
            image = image.cuda()
            channels = channels.cuda()
            meta['landmarks_per_annotator'] = meta['landmarks_per_annotator'].cuda()
            meta['pixel_size'] = meta['pixel_size'].cuda()

            output = model(image.float())
            output = final_layer(output)
            loss = loss_function(output, channels)
            losses.append(loss.item())

            predicted_points, target_points, eres, modes, scaled_predicted_points, scaled_target_points \
                = get_predicted_and_target_points(output, meta['landmarks_per_annotator'], meta['pixel_size'])
            model_predicted_points.append(predicted_points)
            dataset_target_points.append(target_points)
            model_predicted_scaled_points.append(scaled_predicted_points)
            dataset_target_scaled_points.append(scaled_target_points)
            model_eres.append(eres)
            model_modes.append(modes)

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
        model_modes = torch.cat(model_modes)
        # D = Dataset size
        # predicted_points has size [D, N, 2]
        # eres has size [D, N]
        # target_points has size [D, N, 2]

        predicted_points_per_model.append(model_predicted_points)
        scaled_predicted_points_per_model.append(model_predicted_scaled_points)
        eres_per_model.append(model_eres)
        modes_per_model.append(model_modes)

    # predicted_points_per_model is size [M, D, N, 2]
    # eres_per_model is size [M, D, N]
    # target_points is size [D, N, 2]
    predicted_points_per_model = torch.stack(predicted_points_per_model).float()
    scaled_predicted_points_per_model = torch.stack(scaled_predicted_points_per_model).float()
    dataset_target_points = dataset_target_points.float()
    dataset_target_scaled_points = dataset_target_scaled_points.float()
    eres_per_model = torch.stack(eres_per_model).float()
    modes_per_model = torch.stack(modes_per_model).float()

    aggregated_point_dict = use_aggregate_methods(predicted_points_per_model, eres_per_model,
                                                  aggregate_methods=cfg_validation.AGGREGATION_METHODS)
    aggregated_points = aggregated_point_dict[cfg_validation.SDR_AGGREGATION_METHOD]
    aggregated_scaled_point_dict = use_aggregate_methods(scaled_predicted_points_per_model, eres_per_model,
                                                  aggregate_methods=cfg_validation.AGGREGATION_METHODS)
    aggregated_scaled_points = aggregated_scaled_point_dict[cfg_validation.SDR_AGGREGATION_METHOD]

    radial_errors = cal_radial_errors(aggregated_scaled_points, dataset_target_scaled_points)

    measurements_dict = {}
    for measurement in cfg_validation.MEASUREMENTS:
        measurements_dict[measurement] = []

    # quick pass through the images at the end
    # This loop is for per image information and saving images
    for idx, (image, _, meta) in enumerate(loader):

        radial_errors_idx = radial_errors[idx]
        target_points_idx = dataset_target_points[idx]
        aggregated_points_idx = aggregated_points[idx]

        b = 0

        name = meta['file_name'][b]
        txt = "[{}/{}] {}:\t".format(idx + 1, len(loader), name)
        for err in radial_errors_idx:
            txt += "{:.3f}\t".format(err.item())
        txt += "Avg: {:.3f}\t".format(torch.mean(radial_errors_idx).item())

        for measurement in cfg_validation.MEASUREMENTS:
            predicted_angle, target_angle, dif = measure(aggregated_points_idx, target_points_idx,
                                cfg_validation.MEASUREMENTS_SUFFIX, measurement)
            measurements_dict[measurement].append([predicted_angle, target_angle])
            txt += "{}: {:.3f}\t".format(measurement, dif)

        if not training_mode:
            logger.info(txt)

        # display visuals
        for visual_name in visuals:
            image_name = meta["file_name"][b]
            figure_save_path = os.path.join(save_path, visual_name,
                                            "{}_{}".format(image_name, visual_name))
            #txt = "Saving Images in {} for {}".format(figure_save_path, visual_name)
            #logger.info(txt)
            final_figure(image[b], aggregated_points_idx.detach().cpu().numpy(),
                             aggregated_point_dict, target_points_idx.detach().cpu().numpy(),
                             cfg_validation.MEASUREMENTS_SUFFIX, visual_name,
                             save=True, save_path=figure_save_path)

    # Write where images have been saved
    for visual_name in visuals:
        figure_save_path = os.path.join(save_path, visual_name)
        txt = "Saved Images in {} for {}".format(figure_save_path, visual_name)
        logger.info(txt)

    # Overall Statistics
    logger.info("\n-----------Final Statistics-----------")

    # loss
    average_loss = sum(losses) / len(losses)
    txt = "Average loss: {:.3f}".format(average_loss)
    logger.info(txt)

    # Print average landmark localisations
    txt = "Landmark Localisations:\t"
    avg_per_landmark = torch.mean(radial_errors, dim=0)
    std_per_landmark = torch.std(radial_errors, dim=0)
    median_per_landmark = torch.median(radial_errors, dim=0)[0]
    for avg_for_landmark, std_for_landmark, median_for_landmark \
            in zip(avg_per_landmark, std_per_landmark, median_per_landmark):
        txt += "[MEAN: {:.3f}\u00B1{:.3f}, MED: {:.3f}]\t".format(avg_for_landmark.item(),
                                                                  std_for_landmark.item(),
                                                                  median_for_landmark.item())
    overall_avg = torch.mean(radial_errors).item()
    overall_std = torch.std(radial_errors).item()
    overall_med = torch.median(radial_errors).item()
    txt += "[MEAN: {:.3f}\u00B1{:.3f}, MED: {:.3f}]\t".format(overall_avg, overall_std, overall_med)

    for measurement in cfg_validation.MEASUREMENTS:
        measurements_dict[measurement] = torch.Tensor(measurements_dict[measurement])
        avg, std, med, icc = get_stats(measurements_dict[measurement][:, 0], measurements_dict[measurement][:, 1])
        txt += "{}: [MEAN: {:.3f}\u00B1{:.3f}, MED: {:.3f}, ICC: {:.3f}]\t".format(measurement, avg, std, med, icc)

    if not temperature_scaling_mode:
        logger.info(txt)

    sdr_rates = get_sdr_statistics(radial_errors, cfg_validation.SDR_THRESHOLDS)
    txt = "Successful Detection Rates are: "
    for sdr_rate in sdr_rates:
        txt += "{:.2f}%\t".format(sdr_rate)

    if not temperature_scaling_mode:
        logger.info(txt)

    # Final graphics
    if not training_mode:
        # Run the diagnosis experiments
        # I need to find the predicted points and the ground truth points
        # aggregated_scaled_points, dataset_target_scaled_points
        for diagnosis in cfg_validation.DIAGNOSES:
            n, tn, fp, fn, tp, precision, recall, accuracy = diagnose_set(aggregated_scaled_points,
                                                                          dataset_target_scaled_points,
                                                                          cfg_validation.MEASUREMENTS_SUFFIX,
                                                                          diagnosis)

            txt = "Results for {} are n: {}, tn: {}, fp: {}, fn: {}, tp: {}, " \
                  "precision: {:.3f}, recall: {:.3f}, accuracy: {:.3f}".format(
                   diagnosis, n, tn, fp, fn, tp, precision, recall, accuracy)
            logger.info(txt)

        radial_errors_np = radial_errors.detach().cpu().numpy()
        eres_np = eres_per_model[0].detach().cpu().numpy()
        confidence_np = modes_per_model[0].detach().cpu().numpy()

        figure_save_path = os.path.join(save_path, "box_plot")
        display_box_plot(radial_errors_np, figure_save_path)
        logger.info("Saving Box Plot to {}".format(figure_save_path))

        # Save the heatmap analysis plots
        figure_save_path = os.path.join(save_path, "correlation_plot")
        radial_error_vs_ere_graph(radial_errors_np.flatten(), eres_np.flatten(), figure_save_path)
        logger.info("Saving Correlation Plot to {}".format(figure_save_path))

        # Save the heatmap analysis plots
        figure_save_path = os.path.join(save_path, "correlation_plot_2")
        radial_error_vs_ere_graph(radial_errors_np.flatten(), confidence_np.flatten(), figure_save_path)
        logger.info("Saving Correlation Plot 2 to {}".format(figure_save_path))

        # Save the heatmap analysis plots
        figure_save_path = os.path.join(save_path, "roc_plot")
        roc_outlier_graph(radial_errors_np.flatten(), eres_np.flatten(), figure_save_path)
        logger.info("Saving ROC Plot to {}".format(figure_save_path))

        # Save the reliability diagram
        figure_save_path = os.path.join(save_path, "reliability_plot")
        reliability_diagram(radial_errors_np.flatten(), confidence_np.flatten(), figure_save_path)
        logger.info("Saving ROC Plot to {}".format(figure_save_path))

    if temperature_scaling_mode:

        radial_errors_np = radial_errors.detach().cpu().numpy()
        confidence_np = modes_per_model[0].detach().cpu().numpy()
        eres_np = eres_per_model[0].detach().cpu().numpy()

        ece = reliability_diagram(radial_errors_np.flatten(), confidence_np.flatten(), "", save=False)
        logger.info("ECE: {:.3f}".format(ece))

        radial_error_vs_ere_graph(radial_errors_np.flatten(), eres_np.flatten(), "", save=False)
        logger.info("Correlation: {:.3f}".format(ece))

        temperatures = ensemble[0].temperatures
        logger.info("Temperature values: {}".format(temperatures))

    return average_loss, overall_avg
