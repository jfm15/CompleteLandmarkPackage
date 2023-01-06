import torch
import numpy as np


def train_ensemble(ensemble, optimizers, schedulers, training_loader, final_layer, loss_function, logger):

    training_losses = []

    for model_idx in range(len(ensemble)):
        logger.info('-----------Training Model {}-----------'.format(model_idx))

        our_model = ensemble[model_idx]
        our_model = our_model.cuda()
        training_loss = train_model(our_model, final_layer, optimizers[model_idx], schedulers[model_idx],
                                    training_loader, loss_function, logger)
        training_losses.append(training_loss)

        # move model back to cpu
        our_model.cpu()

    return np.mean(training_losses)


def temperature_scale(our_model, optimizer, scheduler, training_loader, final_layer, loss_function, logger):

    logger.info('-----------Fine Tuning Model-----------')

    our_model = our_model.cuda()
    train_model(our_model, final_layer, optimizer, scheduler, training_loader,
                loss_function, logger, temperature_scaling=True)

    # move model back to cpu
    our_model.cpu()


def train_model(model, final_layer, optimizer, scheduler, loader, loss_function, logger, temperature_scaling=False):

    model.train()
    losses_per_epoch = []

    for batch, (image, channels, meta) in enumerate(loader):

        # Put image and channels onto gpu
        image = image.cuda()
        channels = channels.cuda()

        if temperature_scaling:
            with torch.no_grad():
                output = model(image.float())
            output = model.scale(output)
        else:
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

    return np.mean(losses_per_epoch)

