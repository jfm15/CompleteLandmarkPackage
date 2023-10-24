import torch

import numpy as np
import torch.nn as nn
import segmentation_models_pytorch as smp


class UnetPlusPlus(nn.Module):
    def __init__(self, cfg_model, no_of_landmarks):
        super(UnetPlusPlus, self).__init__()
        self.unet = smp.UnetPlusPlus(
            encoder_name=cfg_model.ENCODER_NAME,
            encoder_weights=cfg_model.ENCODER_WEIGHTS,
            decoder_channels=cfg_model.DECODER_CHANNELS,
            decoder_use_batchnorm=cfg_model.BATCH_NORM_DECODER,
            in_channels=cfg_model.IN_CHANNELS,
            classes=no_of_landmarks,
        )
        self.temperatures = nn.Parameter(torch.ones(1, no_of_landmarks, 1, 1), requires_grad=False)

    def forward(self, x):
        return self.unet(x)

    def scale(self, x):
        y = x / self.temperatures
        return y


class Unet(nn.Module):
    def __init__(self, cfg_model, no_of_landmarks):
        super(Unet, self).__init__()
        self.unet = smp.Unet(
            encoder_name=cfg_model.ENCODER_NAME,
            encoder_weights=cfg_model.ENCODER_WEIGHTS,
            decoder_channels=cfg_model.DECODER_CHANNELS,
            in_channels=cfg_model.IN_CHANNELS,
            classes=no_of_landmarks,
        )
        self.temperatures = nn.Parameter(torch.ones(1, no_of_landmarks, 1, 1), requires_grad=False)

    def forward(self, x):
        return self.unet(x)

    def scale(self, x):
        y = x / self.temperatures
        return y


def two_d_softmax(x):
    exp_y = torch.exp(x)
    return exp_y / torch.sum(exp_y, dim=(2, 3), keepdim=True)


def logits(x):
    return x


# dimensions are [B, C, W, H]
# the log is done within this lsos function whereas normally it would be a log softmax
def nll_across_batch(output, target):
    print('nnl',output.shape)
    print('nnl',target.shape)
    nll = target * torch.log(output.double())
    return -torch.mean(torch.sum(nll, dim=(2, 3)))


def bce_across_batch(output, target):
    bce = target * torch.log(output.double()) + (1 - target) * torch.log(1 - output.double())
    return -torch.mean(torch.sum(bce, dim=(2, 3)))


def mse_across_batch(output, target):
    mse = torch.pow(target - output.double(), 2)
    return -torch.mean(torch.sum(mse, dim=(2, 3)))