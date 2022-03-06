import torch

import numpy as np
import torch.nn as nn
import segmentation_models_pytorch as smp


class UnetPlusPlus(nn.Module):
    def __init__(self, cfg_model, no_of_landmarks):
        super(UnetPlusPlus, self).__init__()
        self.unetPlusPlus = smp.UnetPlusPlus(
            encoder_name=cfg_model.ENCODER_NAME,
            encoder_weights=cfg_model.ENCODER_WEIGHTS,
            decoder_channels=cfg_model.DECODER_CHANNELS,
            in_channels=cfg_model.IN_CHANNELS,
            classes=no_of_landmarks,
        )

    def forward(self, x):
        return self.unetPlusPlus(x)


def two_d_softmax(x):
    exp_y = torch.exp(x)
    return exp_y / torch.sum(exp_y, dim=(2, 3), keepdim=True)


def nll_across_batch(output, target):
    nll = -target * torch.log(output.double())
    return torch.mean(torch.sum(nll, dim=(2, 3)))