import torch

import lib.core.validate_gpu as validate_gpu
import lib.core.validate_cpu as validate_cpu


def validate(cfg, ensemble, validation_set_paths, loaders, visuals,
             logger, print_progress=False, image_save_path=None):
    """

    :param cfg: the config file
    :param ensemble: a list of models
    :param loaders: a list of loaders for the images and ground truth
    :param validation_set_paths: a list of directories which the loaders apply to
    :param visuals: the codes for the generic visuals we would like saved
    :param logger: where to print
    :param print_progress: print the X/N line
    :param image_save_path: where to save the images
    :return:
    """

    validate_file = "validate_gpu" if torch.cuda.is_available() else "validate_cpu"

    # This loops over the validation sets
    for loader, validation_images_path in zip(loaders, validation_set_paths):
        logger.info("\n-----------Validating over {}-----------".format(validation_images_path))

        eval("{}.validate_over_set".format(validate_file))\
            (ensemble, loader, visuals, cfg.VALIDATION, print_progress=print_progress, logger=logger)
