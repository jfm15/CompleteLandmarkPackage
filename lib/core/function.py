import torch
import numpy as np


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

