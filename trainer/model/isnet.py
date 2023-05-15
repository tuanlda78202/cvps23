from trainer.base.base_rsu import _up_same, _size_map, RSU, RBC
from trainer.metrics.loss import *
from trainer.model.u2net import U2Net
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MaxPool2d, Conv2d
import sys
import os
import numpy as np
import torch

sys.path.append(os.getcwd())


class ISNetGTEncoder(U2Net):
    def __init__(self, config, out_channel=1, in_channel=1):
        super().__init__(config, out_channel)
        self.in_channel = in_channel
        self.in_conv = RBC(in_ch=in_channel, out_ch=16, stride=2)

    def _make_layers(self, config):
        # Height of RSU Block
        self.height = int(len(config))

        self.add_module(
            "down_sample", MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        )

        # Built RSU Block & Side Layer
        for key, value in config.items():
            self.add_module(key, RSU(value[0], *value[1]))

            self.add_module(
                f"side{value[0][-1]}",
                Conv2d(value[2], self.out_channel, kernel_size=3, padding=1),
            )

    def forward(self, x):
        x = self.in_conv(x)

        def gte(x, height=1):
            x = getattr(self, f"stage{height}")(x)
            x = getattr(self, f"side{height}")(x)

            x_next = gte(getattr(self, "down_sample")(x), height + 1)


class ISNetDIS(U2Net):
    def __init__(self, config, out_channel):
        super().__init__(config, out_channel)


def isnet_gte():
    config = {
        # Config for building RSUs and sides
        # {stage : [name, (height(L), in_ch, mid_ch, out_ch, dilated(optional)), side]}
        "stage1": ["En_1", (7, 16, 16, 64), 64],
        "stage2": ["En_2", (6, 64, 16, 64), 64],
        "stage3": ["En_3", (5, 64, 32, 128), 128],
        "stage4": ["En_4", (4, 128, 32, 256), 256],
        "stage5": ["En_5", (4, 256, 64, 512, True), 512],
        "stage6": ["En_6", (4, 512, 64, 512, True), 512],
    }

    return ISNetGTEncoder(config=config, in_channel=1, out_channel=1)


def isnet():
    config = {
        # Config for building RSUs and sides
        # {stage : [name, (height(L), in_ch, mid_ch, out_ch, dilated), side]}
        "stage1": ["En_1", (7, 64, 32, 64), -1],
        "stage2": ["En_2", (6, 64, 32, 128), -1],
        "stage3": ["En_3", (5, 128, 64, 256), -1],
        "stage4": ["En_4", (4, 256, 128, 512), -1],
        "stage5": ["En_5", (4, 512, 256, 512, True), -1],
        "stage6": ["En_6", (4, 512, 256, 512, True), 512],
        "stage5d": ["De_5", (4, 1024, 256, 512, True), 512],
        "stage4d": ["De_4", (4, 1024, 128, 256), 256],
        "stage3d": ["De_3", (5, 512, 64, 128), 128],
        "stage2d": ["De_2", (6, 256, 32, 64), 64],
        "stage1d": ["De_1", (7, 128, 16, 64), 64],
    }
    return ISNetDIS(config=config, out_channel=1)


def unet(x, height=1):
    if height < 6:
        x1 = getattr(self, f"stage{height}")(x)

        x2 = unet(getattr(self, "down_sample")(x1), height + 1)

        x = getattr(self, f"stage{height}d")(torch.cat((x2, x1), 1))

        side(x, height)

        return _up_same(x, sizes[height - 1]) if height > 1 else x

    else:
        x = getattr(self, f"stage{height}")(x)
        side(x, height)

        return _up_same(x, sizes[height - 1])
