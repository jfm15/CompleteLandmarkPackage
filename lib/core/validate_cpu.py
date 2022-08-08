import torch

from lib.models import two_d_softmax
from lib.core.evaluate import cal_radial_errors
from lib.core.evaluate import use_aggregate_methods
from lib.core.evaluate import get_predicted_and_target_points
from lib.visualisations import figure
from lib.measures import measure


def validate_over_set(ensemble, loader, visuals, cfg_validation, print_progress=False, logger=None):

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
        avg_radial_error = torch.mean(radial_errors)

        name = meta['file_name'][b]
        txt = "[{}/{}] {}:\t".format(idx + 1, len(loader), name)
        for err in radial_errors:
            txt += "{:.2f}\t".format(err.item())
        txt += "Avg: {:.2f}\t".format(avg_radial_error.item())

        for measurement in cfg_validation.MEASUREMENTS:
            _, _, dif = measure(aggregated_points[b], target_points[b],
                                cfg_validation.MEASUREMENTS_SUFFIX, measurement)
            txt += "{}: {:.2f}\t".format(measurement, dif)

        logger.info(txt)

        # TODO: If singular experiment print out heatmaps and eres

        for visual_name in visuals:
            figure(image[b].detach().numpy(), aggregated_points[b].detach().numpy(),
                   target_points[b].detach().numpy(), cfg_validation.MEASUREMENTS_SUFFIX, visual_name)
