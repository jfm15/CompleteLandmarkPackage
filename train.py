import argparse
import torch
import os

import model

import numpy as np
import matplotlib.pyplot as plt

from model import two_d_softmax
from model import nll_across_batch
from landmark_dataset import LandmarkDataset
from utils import prepare_for_training
from function import train_model
from test import print_validation_of_ensemble
from torchsummary.torchsummary import summary_string


'''
Code design based on Bin Xiao's Deep High Resolution Network Repository:
https://github.com/leoxiaobin/deep-high-resolution-net.pytorch
'''


def parse_args():
    parser = argparse.ArgumentParser(description='Train a network to detect landmarks')

    parser.add_argument('--cfg',
                        help='The path to the configuration file for the experiment',
                        required=True,
                        type=str)

    parser.add_argument('--training_images',
                        help='The path to the training images',
                        type=str,
                        required=True,
                        default='')

    parser.add_argument('--validation_images',
                        help='The path to the validation images',
                        type=str,
                        nargs='+',
                        required=True,
                        default='')

    parser.add_argument('--annotations',
                        help='The path to the directory where annotations are stored',
                        type=str,
                        required=True,
                        default='')

    parser.add_argument('--output_path',
                        help='The path to the directory where annotations are stored',
                        type=str,
                        required=True,
                        default='')

    parser.add_argument('-d', '--debug',
                        action='store_true',
                        help='show passing the training examples')

    args = parser.parse_args()

    return args


def main():
    # get arguments and the experiment file
    args = parse_args()

    cfg, logger, output_path, yaml_file_name = prepare_for_training(args.cfg, args.output_path)

    # print the arguments into the log
    logger.info("-----------Arguments-----------")
    logger.info(vars(args))
    logger.info("")

    # print the configuration into the log
    logger.info("-----------Configuration-----------")
    logger.info(cfg)
    logger.info("")

    training_dataset = LandmarkDataset(args.training_images, args.annotations, cfg.DATASET, perform_augmentation=True,
                                       subset=("below", cfg.TRAIN.LABELED_SUBSET))
    training_loader = torch.utils.data.DataLoader(training_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True)

    validation_datasets = []
    validation_loaders = []
    for validation_images_path in args.validation_images:
        validation_dataset = LandmarkDataset(validation_images_path, args.annotations, cfg.DATASET, gaussian=False,
                                         perform_augmentation=False)
        validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=1, shuffle=False)
        validation_datasets.append(validation_dataset)
        validation_loaders.append(validation_loader)

    # Used for debugging
    if args.debug:
        for batch, (image, channels, meta) in enumerate(training_loader):
            print(meta["file_name"])
            plt.imshow(np.moveaxis(image[0].detach().cpu().numpy(), 0, -1), cmap='gray')
            squashed_channels = np.max(channels[0].detach().cpu().numpy(), axis=0)
            # squashed_channels = channels[0].detach().cpu().numpy()[8]
            plt.imshow(squashed_channels, cmap='inferno', alpha=0.5)
            landmarks_per_annotator = meta['landmarks_per_annotator'].detach().cpu().numpy()
            avg_key_point_locations = np.mean(landmarks_per_annotator[0], axis=0)
            plt.axis("off")
            for i, positions in enumerate(avg_key_point_locations):
                plt.text(positions[0], positions[1], "{}".format(i + 1), color="yellow", fontsize="small")
            plt.show()

    for run in range(cfg.TRAIN.REPEATS):

        ensemble = []
        optimizers = []
        schedulers = []

        for _ in range(cfg.TRAIN.ENSEMBLE_MODELS):
            this_model = eval("model." + cfg.MODEL.NAME)(cfg.MODEL, cfg.DATASET.KEY_POINTS)
            optimizer = torch.optim.Adam(this_model.parameters(), lr=cfg.TRAIN.LR)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[4, 6, 8], gamma=0.1)
            ensemble.append(this_model)
            optimizers.append(optimizer)
            schedulers.append(scheduler)

        if run == 0:
            logger.info("-----------Model Summary-----------")
            model_summary, _ = summary_string(ensemble[0], (1, *cfg.DATASET.CACHED_IMAGE_SIZE), device=torch.device('cpu'))
            logger.info(model_summary)

        logger.info("-----------Experiment {}-----------".format(run + 1))

        for epoch in range(cfg.TRAIN.EPOCHS):

            logger.info('-----------Epoch {} Supervised Training-----------'.format(epoch))

            for model_idx in range(len(ensemble)):

                logger.info('-----------Training Model {}-----------'.format(model_idx))

                our_model = ensemble[model_idx]
                our_model = our_model.cuda()
                train_model(our_model, two_d_softmax, optimizers[model_idx], schedulers[model_idx], training_loader,
                            nll_across_batch, logger)

                # move model back to cpu
                our_model.cpu()

        # Validate
        print_validation_of_ensemble(cfg, ensemble, args.validation_images, validation_loaders, logger)

        logger.info('-----------Saving Models-----------')
        model_run_path = os.path.join(output_path, "run:{}_models".format(run))
        if not os.path.exists(model_run_path):
            os.makedirs(model_run_path)
        for model_idx in range(len(ensemble)):
            our_model = ensemble[model_idx]
            save_model_path = os.path.join(model_run_path, "{}_model_run:{}_idx:{}.pth".format(yaml_file_name, run, model_idx))
            logger.info("Saving Model {}'s State Dict to {}".format(model_idx, save_model_path))
            torch.save(our_model.state_dict(), save_model_path)


if __name__ == '__main__':
    main()