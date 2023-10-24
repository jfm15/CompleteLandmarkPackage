import argparse
import torch
import os
import math

import _init_paths
import lib
import lib.models
import numpy as np

from lib.dataset import LandmarkDataset
from lib.utils import prepare_for_training
from lib.core.function import train_ensemble
from lib.visualisations import preliminary_figure
#from torchsummary.torchsummary import summary_string

import lib.core.validate_cpu as validate_cpu
import lib.core.validate_gpu as validate_gpu


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
    torch.cuda.empty_cache()
    
    cfg, logger, output_path, yaml_file_name = prepare_for_training(args.cfg, args.output_path)

    # print the arguments into the log
    logger.info("-----------Arguments-----------")
    logger.info(vars(args))
    logger.info("")

    # print the configuration into the log
    logger.info("-----------Configuration-----------")
    logger.info(cfg)
    logger.info("")

    training_dataset = LandmarkDataset(args.images, args.annotations, cfg.DATASET, perform_augmentation=True,
                                       subset=("below", cfg.TRAIN.LABELED_SUBSET), partition=args.partition,
                                       partition_label="training")
    training_loader = torch.utils.data.DataLoader(training_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True)
    validation_dataset = LandmarkDataset(args.images, args.annotations, cfg.DATASET, gaussian=False,
                                         perform_augmentation=False, partition=args.partition,
                                         partition_label="validation")
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=1, shuffle=False)

    test_dataset = LandmarkDataset(args.images, args.annotations, cfg.DATASET, gaussian=False,
                                   perform_augmentation=False, partition=args.partition,
                                   partition_label="testing")
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Used for debugging
    if args.debug:
        for batch, (image, channels, meta) in enumerate(training_loader):
            print(meta["file_name"])
            landmarks_per_annotator = meta['landmarks_per_annotator']
            target_points = torch.mean(landmarks_per_annotator[0], dim=0)
            preliminary_figure(image[0], channels[0].detach().cpu().numpy(), target_points, "show_channels", output_path+'/output/debug')

    if torch.cuda.is_available():
        validate_file = "validate_gpu"
    else:
        validate_file = "validate_cpu"
    validate_file = "validate_cpu"
    
    final_layer = eval("lib.models." + cfg.TRAIN.FINAL_LAYER)
    loss_function = eval("lib.models." + cfg.TRAIN.LOSS_FUNCTION)

    for run in range(cfg.TRAIN.REPEATS):

        ensemble = []
        optimizers = []
        schedulers = []

        for _ in range(cfg.TRAIN.ENSEMBLE_MODELS):
            this_model = eval("lib.models." + cfg.MODEL.NAME)(cfg.MODEL, cfg.DATASET.KEY_POINTS)
            optimizer = torch.optim.Adam(this_model.parameters(), lr=cfg.TRAIN.LR)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 15], gamma=0.1)
            ensemble.append(this_model)
            optimizers.append(optimizer)
            schedulers.append(scheduler)

        if run == 0:
            logger.info("-----------Model Summary-----------")
            #model_summary, _ = summary_string(ensemble[0], (1, *cfg.DATASET.CACHED_IMAGE_SIZE), device=torch.device('cpu'))
            #logger.info(model_summary)

        logger.info("-----------Experiment {}-----------".format(run + 1))

        best_mre = math.inf
        best_state_dicts = []
        no_better_count = 0
        
        for epoch in range(cfg.TRAIN.EPOCHS):

            logger.info('-----------Epoch {} Supervised Training-----------'.format(epoch))
            train_ensemble(ensemble, optimizers, schedulers, training_loader, final_layer, loss_function, logger)

            # Validate
            with torch.no_grad():
                logger.info('-----------Validation Set-----------')
                save_path = os.path.join(output_path)

                _, current_mre = eval("{}.validate_over_set".format(validate_file)) \
                    (ensemble, validation_loader, final_layer, loss_function, [], cfg.VALIDATION, save_path,
                     logger=logger, training_mode=True)

                if current_mre < best_mre:
                    logger.info('-----------Best Results So Far-----------')
                    # keep track of best models
                    no_better_count = 0
                    best_mre = current_mre
                    ensemble_state_dict = []
                    for model_idx in range(len(ensemble)):
                        ensemble_state_dict.append(ensemble[model_idx].state_dict())
                    best_state_dicts = ensemble_state_dict
                else:
                    no_better_count += 1

            if no_better_count == cfg.TRAIN.EARLY_STOPPING:
                logger.info('-----------Halting Training As No Improvment Seen For {} Epochs-----------'
                            .format(no_better_count))
                break

        # reload best models
        for model_idx in range(len(ensemble)):
            our_model = ensemble[model_idx]
            our_model.load_state_dict(best_state_dicts[model_idx], strict=True)

        with torch.no_grad():
            logger.info('-----------Test Set-----------')
            save_fig_path = os.path.join(output_path, "eval_img_train_run{}".format(run))
            if not os.path.exists(save_fig_path):
                os.makedirs(save_fig_path)
            eval("{}.validate_over_set".format(validate_file)) \
                    (ensemble, test_loader, final_layer, loss_function, [], cfg.VALIDATION, save_fig_path,
                 logger=logger, training_mode=False)

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