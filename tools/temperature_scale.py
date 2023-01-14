import argparse
import torch
import wandb
import os

import _init_paths
import lib
import lib.models

from lib.dataset import LandmarkDataset
from lib.utils import prepare_for_testing
from lib.core.function import temperature_scale
from torchsummary.torchsummary import summary_string

import lib.core.validate_cpu as validate_cpu
import lib.core.validate_gpu as validate_gpu


def parse_args():
    parser = argparse.ArgumentParser(description='Train the temperature scaling parameters')

    parser.add_argument('--cfg',
                        help='The path to the configuration file for the experiment',
                        required=True,
                        type=str)

    parser.add_argument('--images',
                        help='The path to the training images',
                        type=str,
                        required=True,
                        default='')

    parser.add_argument('--annotations',
                        help='The path to the directory where annotations are stored',
                        type=str,
                        required=True,
                        default='')

    parser.add_argument('--partition',
                        help='The path to the partition file',
                        type=str,
                        required=True,
                        default='')

    parser.add_argument('--pretrained_model',
                        help='the path to a pretrained model',
                        type=str,
                        required=True)

    parser.add_argument('--tags',
                        help='tags which are passed to weights and biases',
                        nargs='+',
                        required=False,
                        default=[])

    args = parser.parse_args()

    return args


def validate(model, validation_loader, final_layer, loss_function, cfg_validation,
             save_path, logger, epoch=0, save_graphs=False):

    if torch.cuda.is_available():
        validate_file = "validate_gpu"
    else:
        validate_file = "validate_cpu"

    # Validate
    with torch.no_grad():
        logger.info('-----------Validation Set-----------')

        if not save_graphs:
            eval("{}.validate_over_set".format(validate_file))\
                ([model], validation_loader, final_layer, loss_function, [], cfg_validation, save_path,
                 logger=logger, training_mode=True, temperature_scaling_mode=True, epoch=epoch)
        else:
            eval("{}.validate_over_set".format(validate_file)) \
                ([model], validation_loader, final_layer, loss_function, [], cfg_validation, save_path,
                 logger=logger)


def main():

    # Get arguments and the experiment file
    args = parse_args()

    cfg, logger, output_path, yaml_file_name = prepare_for_testing(args.cfg, args.pretrained_model)

    wandb.login(key="f6e720fe9b2f70bdd25b65e68e51d5163e2b0337")

    tags = ['scaling'] + args.tags
    wandb.init(project="complete_landmark_package", name=yaml_file_name, config=cfg,
               entity="j-mccouat", tags=tags)

    # Print the arguments into the log
    logger.info("-----------Arguments-----------")
    logger.info(vars(args))
    logger.info("")

    # Print the configuration into the log
    logger.info("-----------Configuration-----------")
    logger.info(cfg)
    logger.info("")

    training_dataset = LandmarkDataset(args.images, args.annotations, cfg.DATASET, perform_augmentation=True,
                                       partition=args.partition, partition_label="training")
    training_loader = torch.utils.data.DataLoader(training_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True)

    validation_dataset = LandmarkDataset(args.images, args.annotations, cfg.DATASET, gaussian=False,
                                         perform_augmentation=False, partition=args.partition,
                                         partition_label="validation")
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=1, shuffle=False)

    # Load model and state dict from file
    our_model = eval("lib.models." + cfg.MODEL.NAME)(cfg.MODEL, cfg.DATASET.KEY_POINTS)
    loaded_state_dict = torch.load(args.pretrained_model)
    our_model.load_state_dict(loaded_state_dict, strict=True)

    logger.info("-----------Model Summary-----------")
    model_summary, _ = summary_string(our_model, (1, *cfg.DATASET.CACHED_IMAGE_SIZE), device=torch.device('cpu'))
    logger.info(model_summary)

    our_model.temperatures.requires_grad = True
    optimizer = torch.optim.SGD([our_model.temperatures], lr=cfg.TRAIN.SCALING_LR)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[21], gamma=0.1)
    final_layer = eval("lib.models." + cfg.TRAIN.FINAL_LAYER)
    loss_function = eval("lib.models." + cfg.TRAIN.LOSS_FUNCTION)

    image_save_path = os.path.join(output_path, 'images')
    if not os.path.exists(image_save_path):
        os.makedirs(image_save_path)

    validate(our_model, validation_loader, final_layer, loss_function,
             cfg.VALIDATION, image_save_path, logger, save_graphs=True)

    for epoch in range(20):

        logger.info('-----------Epoch {} Temperature Scaling-----------'.format(epoch))
        temperature_scale(our_model, optimizer, scheduler, training_loader, final_layer, loss_function, logger)

        validate(our_model, validation_loader, final_layer, loss_function,
                 cfg.VALIDATION, image_save_path, logger, epoch=epoch)

    logger.info("-----------Temperature Scaling Complete-----------")
    validate(our_model, validation_loader, final_layer, loss_function,
             cfg.VALIDATION, image_save_path, logger, save_graphs=True)


if __name__ == '__main__':
    main()
