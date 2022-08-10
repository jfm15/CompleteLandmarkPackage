import torch
import numpy as np

from lib.models import two_d_softmax
from lib.models import nll_across_batch


def train_ensemble(ensemble, optimizers, schedulers, training_loader, epochs, logger):

    for epoch in range(epochs):

        logger.info('-----------Epoch {} Supervised Training-----------'.format(epoch))

        for model_idx in range(len(ensemble)):
            logger.info('-----------Training Model {}-----------'.format(model_idx))

            our_model = ensemble[model_idx]
            our_model = our_model.cuda()
            train_model(our_model, two_d_softmax, optimizers[model_idx], schedulers[model_idx], training_loader,
                        nll_across_batch, logger)

            # move model back to cpu
            our_model.cpu()


def train_model(model, final_layer, optimizer, scheduler, loader, loss_function, logger):
    model.train()
    losses_per_epoch = []

    for batch, (image, channels, meta) in enumerate(loader):

        # Put image and channels onto gpu
        image = image.cuda()
        channels = channels.cuda()

        output = model(image.float())
        output = final_layer(output)

        optimizer.zero_grad()
        loss = loss_function(output, channels)
        loss.backward()

        optimizer.step()

        losses_per_epoch.append(loss.item())

        if (batch + 1) % 5 == 0:
            logger.info("[{}/{}]\tLoss: {:.3f}".format(batch + 1, len(loader), np.mean(losses_per_epoch)))

    scheduler.step()

