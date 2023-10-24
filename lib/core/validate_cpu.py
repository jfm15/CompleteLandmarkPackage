import os
import torch

from lib.core.evaluate import cal_radial_errors
from lib.core.evaluate import use_aggregate_methods
from lib.core.evaluate import get_predicted_and_target_points
from lib.visualisations import intermediate_figure
from lib.visualisations import final_figure
from lib.measures import measure


def validate_over_set(ensemble, loader, final_layer, loss_function, visuals, cfg_validation, save_path, logger=None, training_mode=False):

    all_radial_errors = []
    all_measurement_difs = []
    losses = []

    for idx, (image, channels, meta) in enumerate(loader):

        image_predicted_points = []
        image_scaled_predicted_points = []
        image_eres = []

        for model in ensemble:

            model.eval()
            output = model(image.float())
            output = final_layer(output)
            print('out shape',output.shape)
            loss = loss_function(output, channels)
            losses.append(loss.item())
            predicted_points, target_points, eres, _, scaled_predicted_points, scaled_target_points \
                = get_predicted_and_target_points(output, meta['landmarks_per_annotator'], meta['pixel_size'])
            image_predicted_points.append(predicted_points)
            image_scaled_predicted_points.append(scaled_predicted_points)
            image_eres.append(eres)

            # print figures
            b = 0
            for visual_name in visuals:
                if not os.path.isdir(save_path+'/'+visual_name):
                    os.mkdir(save_path+'/'+visual_name)

                intermediate_figure(image[b], output[b].numpy(), predicted_points[b],
                                    target_points[b], eres[b], visual_name, save=True, save_path=save_path+'/'+visual_name+'/'+meta['file_name'][0])
               
        # put these arrays into a format suitable for the aggregate methods function
        image_predicted_points = torch.unsqueeze(torch.cat(image_predicted_points), 1).float()
        image_scaled_predicted_points = torch.unsqueeze(torch.cat(image_scaled_predicted_points), 1).float()
        
        image_scaled_predicted_points_txt = image_scaled_predicted_points.squeeze().numpy()
        
        if not os.path.isdir(save_path+'/txt/'):
            os.mkdir(save_path+'/txt/')
        
        with open(save_path+'/txt/'+meta['file_name'][0]+".txt", 'a') as output:
            for i in range(len(image_scaled_predicted_points_txt)):
                row=image_scaled_predicted_points_txt[i].tolist()
                data_str = str([round(row[1],5),round(row[0],5)])[1:-1]
                output.write(data_str+"\n")
        
        image_eres = torch.unsqueeze(torch.cat(image_eres), 1).float()

        aggregated_point_dict = use_aggregate_methods(image_predicted_points, image_eres,
                                                      aggregate_methods=cfg_validation.AGGREGATION_METHODS)
        aggregated_points = aggregated_point_dict[cfg_validation.SDR_AGGREGATION_METHOD]

        aggregated_scaled_point_dict = use_aggregate_methods(image_scaled_predicted_points, image_eres,
                                                             aggregate_methods=cfg_validation.AGGREGATION_METHODS)
        aggregated_scaled_points = aggregated_scaled_point_dict[cfg_validation.SDR_AGGREGATION_METHOD]

        # assumes the batch size is 1
        b = 0

        radial_errors = cal_radial_errors(aggregated_scaled_points, scaled_target_points)[b]
        all_radial_errors.append(radial_errors)
        avg_radial_error = torch.mean(radial_errors)

        name = meta['file_name'][b]
        txt = "[{}/{}] {}:\t".format(idx + 1, len(loader), name)
        for err in radial_errors:
            txt += "{:.3f}\t".format(err.item())
        txt += "Avg: {:.3f}\t".format(avg_radial_error.item())

        for measurement in cfg_validation.MEASUREMENTS:
            predicted_angle, target_angle, dif = measure(aggregated_points[b], target_points[b],
                                cfg_validation.MEASUREMENTS_SUFFIX, measurement)
            all_measurement_difs.append(dif)
            txt += "{}: [{:.3f}, {:.3f}, {:.3f}]\t".format(measurement, predicted_angle, target_angle, dif)

        logger.info(txt)
        #visuals = visuals[0].split(',')

        for visual_name in visuals:
            final_figure(image[b], aggregated_points[b],
                         aggregated_point_dict, target_points[b],
                         cfg_validation.MEASUREMENTS_SUFFIX, visual_name,save=True, save_path=save_path+'/'+visual_name+'/'+meta['file_name'][0])

    average_loss = sum(losses) / len(losses)
    txt = "Average loss: {:.3f}".format(average_loss)
    logger.info(txt)

    all_radial_errors = torch.stack(all_radial_errors)
    average_radial_error = torch.mean(all_radial_errors).item()
    std_radial_error = torch.std(all_radial_errors).item()
    txt = "The average landmark localisation error is: {:.3f}\u00B1{:.3f}".format(average_radial_error, std_radial_error)
    logger.info(txt)

    all_measurement_difs = torch.FloatTensor(all_measurement_difs)
    average_measurement_dif = torch.mean(all_measurement_difs)
    std_measurement_dif = torch.std(all_measurement_difs)
    txt = "The average measurement error is: {:.3f}\u00B1{:.3f}".format(average_measurement_dif,
                                                                        std_measurement_dif)
    logger.info(txt)

    return average_loss, average_radial_error
