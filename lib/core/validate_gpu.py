import os
import torch
import wandb

from lib.utils import get_stats
from lib.core.evaluate import cal_radial_errors
from lib.core.evaluate import use_aggregate_methods
from lib.core.evaluate import get_predicted_and_target_points
from lib.core.evaluate import get_threshold_table

from lib.visualisations import intermediate_figure
from lib.visualisations import final_figure
from lib.visualisations import display_box_plot
from lib.visualisations import correlation_graph
from lib.visualisations import reliability_diagram
from lib.visualisations import roc_outlier_graph

from lib.measures import measure
from lib.measures import diagnose_set


def validate_over_set(ensemble, loader, final_layer, loss_function, visuals, cfg_validation, save_path,
                      logger=None, training_mode=False, temperature_scaling_mode=False, epoch=0):

    predicted_points_per_model = []
    eres_per_model = []
    modes_per_model = []
    pixel_size_per_model = []
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
        model_pixel_sizes = []

        for idx, (image, channels, meta) in enumerate(loader):

            # allocate
            image = image.cuda()
            channels = channels.cuda()
            meta['landmarks_per_annotator'] = meta['landmarks_per_annotator'].cuda()
            meta['pixel_size'] = meta['pixel_size'].cuda()

            output = model(image.float())
            output = model.scale(output)
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

            # turn pixel size into the same shape as the ere
            pixel_size_tensor = meta['pixel_size'][0][0]
            pixel_size_tensor = pixel_size_tensor.repeat(eres.size())
            model_pixel_sizes.append(pixel_size_tensor)

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
        model_pixel_sizes = torch.cat(model_pixel_sizes)
        # D = Dataset size
        # predicted_points has size [D, N, 2]
        # eres has size [D, N]
        # target_points has size [D, N, 2]

        predicted_points_per_model.append(model_predicted_points)
        scaled_predicted_points_per_model.append(model_predicted_scaled_points)
        eres_per_model.append(model_eres)
        modes_per_model.append(model_modes)
        pixel_size_per_model.append(model_pixel_sizes)

    # predicted_points_per_model is size [M, D, N, 2]
    # eres_per_model is size [M, D, N]
    # target_points is size [D, N, 2]
    predicted_points_per_model = torch.stack(predicted_points_per_model).float()
    scaled_predicted_points_per_model = torch.stack(scaled_predicted_points_per_model).float()
    dataset_target_points = dataset_target_points.float()
    dataset_target_scaled_points = dataset_target_scaled_points.float()
    eres_per_model = torch.stack(eres_per_model).float()
    modes_per_model = torch.stack(modes_per_model).float()
    pixel_size_per_model = torch.stack(pixel_size_per_model).float()

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

    if not temperature_scaling_mode:
        logger.info(txt)

    # Final graphics
    # We want the following functionality:
    # 1) Add the end of the test script or during the temperature scaling phase where we want things to be logged
    if not training_mode or temperature_scaling_mode:
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
        pixel_size_np = pixel_size_per_model[0].detach().cpu().numpy()

        '''
        figure_save_path = os.path.join(save_path, "box_plot")
        display_box_plot(radial_errors_np, figure_save_path)
        logger.info("Saving Box Plot to {}".format(figure_save_path))
        '''

        # Save the heatmap analysis plots
        radial_ere_crl, radial_ere_wb_img = correlation_graph(radial_errors_np.flatten(), eres_np.flatten(),
                                                              "True Radial error (mm)", "Expected Radial Error (ERE) (mm)")

        # Save the heatmap analysis plots
        radial_cof_crl, radial_conf_wb_img = correlation_graph(radial_errors_np.flatten(), confidence_np.flatten(),
                                                               "True Radial error (mm)",
                                                               "Confidence")

        # Save the heatmap analysis plots
        proposed_threshold, auc, roc_wb_img = roc_outlier_graph(radial_errors_np.flatten(), eres_np.flatten())

        # Save the reliability diagram
        ece, reliability_diagram_wb_image = reliability_diagram(radial_errors_np.flatten(), confidence_np.flatten(),
                                                                pixel_size_np.flatten())

        # data should be 2 rows of mre and sdr scores for the threshold
        threshold_table_data = get_threshold_table(torch.flatten(radial_errors), torch.flatten(eres_per_model[0]),
                                                   proposed_threshold, cfg_validation.SDR_THRESHOLDS)

        if temperature_scaling_mode:
            wandb.log({"radial_ere_cor": radial_ere_crl,
                       "radial_confidence_cor": radial_cof_crl,
                       "auc": auc,
                       "ece": ece,
                       "epoch": epoch})

        else:

            threshold_table = wandb.Table(columns=["Set", "# landmarks", "MRE"] + cfg_validation.SDR_THRESHOLDS,
                                          data=threshold_table_data)

            wandb.log({"threshold_table": threshold_table,
                       "radial_ere_correlation_plot": radial_ere_wb_img,
                       "radial_confidence_correlation_plot": radial_conf_wb_img,
                       "roc_ere_plot": roc_wb_img,
                       "reliability_diagram": reliability_diagram_wb_image})

            wandb.run.summary["radial_ere_correlation"] = radial_ere_crl
            wandb.run.summary["radial_confidence_correlation"] = radial_cof_crl
            wandb.run.summary["auc"] = auc
            wandb.run.summary["ece"] = ece
            wandb.run.summary["MRE"] = overall_avg

    return average_loss, overall_avg
