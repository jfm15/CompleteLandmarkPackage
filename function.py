import torch
import numpy as np
from evaluate import get_predicted_and_target_points
from evaluate import get_hottest_points
from visualise import visualise_heatmaps


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


def use_model(model, final_layer, loader, loss_function,
              logger=None, print_progress=False, print_heatmap_images=False,
              model_idx=0, save_image_path=None):
    model.eval()
    all_losses = []
    all_predicted_points = []
    all_target_points = []
    all_eres = []

    with torch.no_grad():
        for idx, (image, channels, meta) in enumerate(loader):
            # Put image and channels onto gpu
            image = image.cuda()
            channels = channels.cuda()
            meta['landmarks_per_annotator'] = meta['landmarks_per_annotator'].cuda()
            meta['pixel_size'] = meta['pixel_size'].cuda()

            output = model(image.float())
            output = final_layer(output)

            loss = loss_function(output, channels)
            all_losses.append(loss.item())

            predicted_points, target_points, eres \
                = get_predicted_and_target_points(output, meta['landmarks_per_annotator'], meta['pixel_size'])
            all_predicted_points.append(predicted_points)
            all_target_points.append(target_points)
            all_eres.append(eres)
            # predicted_points has size [B, N, 2]
            # target_points has size [B, N, 2]
            # eres has size [B, N]

            predicted_pixel_points = get_hottest_points(output).cpu().detach().numpy()

            if print_progress:
                if (idx + 1) % 30 == 0:
                    logger.info("[{}/{}]".format(idx + 1, len(loader)))

            if print_heatmap_images:
                name = meta['file_name'][0]
                visualise_heatmaps(image.cpu().detach().numpy(),
                                   output.cpu().detach().numpy(),
                                   eres.cpu().detach().numpy(),
                                   predicted_pixel_points,
                                   model_idx, name, save_image_path)

    model.cpu()

    all_predicted_points = torch.cat(all_predicted_points)
    all_target_points = torch.cat(all_target_points)
    all_eres = torch.cat(all_eres)
    # D = Dataset size
    # predicted_points has size [D, N, 2]
    # target_points has size [D, N, 2]
    # eres has size [D, N]

    return all_losses, all_predicted_points, all_target_points, all_eres
