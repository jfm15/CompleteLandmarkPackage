import torch
import numpy as np
import matplotlib.pyplot as plt

from evaluate import get_predicted_and_target_points
from model import two_d_softmax

from general_visuals import singular_only_graphics


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


def use_model(model, final_layer, loader, graphic_codes, logger=None, print_progress=False):
    model.eval()
    all_predicted_points = []
    all_target_points = []
    all_eres = []

    with torch.no_grad():

        # image has shape [batch_size, channels, height, width]
        for idx, (image, _, meta) in enumerate(loader):

            # Put image and channels onto gpu
            # image = image.cuda()
            #meta['landmarks_per_annotator'] = meta['landmarks_per_annotator'].cuda()
            #meta['pixel_size'] = meta['pixel_size'].cuda()

            output = model(image.float())
            output = final_layer(output)

            predicted_points, target_points, eres \
                = get_predicted_and_target_points(output, meta['landmarks_per_annotator'], meta['pixel_size'])
            all_predicted_points.append(predicted_points)
            all_target_points.append(target_points)
            all_eres.append(eres)
            # predicted_points has size [B, N, 2]
            # target_points has size [B, N, 2]
            # eres has size [B, N]

            # the batch size here is 1 so we can reference the first element in the array
            b = 0
            heatmaps = output.detach().numpy()
            for code in graphic_codes:
                singular_only_graphics(image[b], heatmaps[b], predicted_points[b],
                                       target_points[b], eres[b], code)

            if print_progress:
                if (idx + 1) % 30 == 0:
                    logger.info("[{}/{}]".format(idx + 1, len(loader)))

    model.cpu()

    all_predicted_points = torch.cat(all_predicted_points)
    all_target_points = torch.cat(all_target_points)
    all_eres = torch.cat(all_eres)
    # D = Dataset size
    # predicted_points has size [D, N, 2]
    # target_points has size [D, N, 2]
    # eres has size [D, N]

    return all_predicted_points, all_target_points, all_eres

