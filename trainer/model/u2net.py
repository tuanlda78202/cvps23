import numpy as np
import torch
import torch.nn as nn
from torch.nn import MaxPool2d, Conv2d
import torch.nn.functional as F
import sys, os

sys.path.append(os.getcwd())

from trainer.base.base_rsu import _up_same, _size_map, RSU


# U2Net
class U2Net(nn.Module):
    def __init__(self, config, out_channel):
        super().__init__()
        self.out_channel = out_channel
        self._make_layers(config)

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + "\nTrainable parameters: {}".format(params)

    def _make_layers(self, config):
        # Height of RSU Block
        self.height = int((len(config) + 1) / 2)

        # MaxPood 2D
        self.add_module(
            "down_sample", MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        )

        # Built RSU Block & Side Layer
        for key, value in config.items():
            # RSU Block
            self.add_module(key, RSU(value[0], *value[1]))

            # Side Layer
            if value[2] > 0:
                self.add_module(
                    f"side{value[0][-1]}",
                    Conv2d(value[2], self.out_channel, kernel_size=3, padding=1),
                )

        # Fuse Layer
        self.add_module(
            "out_conv2d",
            Conv2d(
                int(self.height * self.out_channel), self.out_channel, kernel_size=1
            ),
        )

    def forward(self, x):
        sizes = _size_map(x, self.height)
        maps = []  # storage for list maps

        # Side 6 saliency maps
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

        def side(x, height):
            # Generates six side output salience probability maps by conv 3x3
            x = getattr(self, f"side{height}")(x)

            # Up-sample the logits (conv outputs before sigmoid functions) of the side output saliency maps
            # to the input image size
            x = _up_same(x, sizes[1])
            maps.append(x)

        # Fuse 6 saliency maps + Outconv
        def fuse(x):
            # Fuse with a concatenation operation followed by a 1×1 convolution layer

            # Reverse for add d0 first
            maps.reverse()
            x = torch.cat(maps, dim=1)

            # x_fuse = con1x1 6 saliency maps
            x_fuse = getattr(self, "out_conv2d")(x)
            maps.insert(0, x_fuse)

            # List d0 & Decoder
            list_maps = [torch.sigmoid(x) for x in maps]

            return x_fuse, list_maps

        # U-Net x
        x_unet = unet(x)

        # X_fuse sigmoid + List feature maps
        x_fuse, list_maps = fuse(x_unet)
        x_fuse = torch.sigmoid(x_fuse)

        # x_map for metrics, list_maps for loss
        return x_fuse, list_maps


def u2net_full():
    full = {
        # Config for building RSUs and sides
        # {stage : [name, (height(L), in_ch, mid_ch, out_ch, dilated), side]}
        "stage1": ["En_1", (7, 3, 32, 64), -1],
        "stage2": ["En_2", (6, 64, 32, 128), -1],
        "stage3": ["En_3", (5, 128, 64, 256), -1],
        "stage4": ["En_4", (4, 256, 128, 512), -1],
        # RSU-4F: “F” for dilated version
        # All intermediate feature maps of RSU-4F have the same resolution with its input feature maps.
        "stage5": ["En_5", (4, 512, 256, 512, True), -1],
        "stage6": ["En_6", (4, 512, 256, 512, True), 512],
        # Each decoder stage takes the concatenation of the up-sampled feature maps from its previous stage
        # and those from its symmetrical encoder stage as the input,
        "stage5d": ["De_5", (4, 1024, 256, 512, True), 512],
        "stage4d": ["De_4", (4, 1024, 128, 256), 256],
        "stage3d": ["De_3", (5, 512, 64, 128), 128],
        "stage2d": ["De_2", (6, 256, 32, 64), 64],
        "stage1d": ["De_1", (7, 128, 16, 64), 64],
    }
    return U2Net(config=full, out_channel=1)


def u2net_lite():
    lite = {
        # Config for building RSUs and sides
        # {stage : [name, (height(L), in_ch, mid_ch, out_ch, dilated), side]}
        "stage1": ["En_1", (7, 3, 16, 64), -1],
        "stage2": ["En_2", (6, 64, 16, 64), -1],
        "stage3": ["En_3", (5, 64, 16, 64), -1],
        "stage4": ["En_4", (4, 64, 16, 64), -1],
        # RSU-4F: “F” for dilated version
        # All intermediate feature maps of RSU-4F have the same resolution with its input feature maps.
        "stage5": ["En_5", (4, 64, 16, 64, True), -1],
        "stage6": ["En_6", (4, 64, 16, 64, True), 64],
        # Each decoder stage takes the concatenation of the up-sampled feature maps from its previous stage
        # and those from its symmetrical encoder stage as the input,
        "stage5d": ["De_5", (4, 128, 16, 64, True), 64],
        "stage4d": ["De_4", (4, 128, 16, 64), 64],
        "stage3d": ["De_3", (5, 128, 16, 64), 64],
        "stage2d": ["De_2", (6, 128, 16, 64), 64],
        "stage1d": ["De_1", (7, 128, 16, 64), 64],
    }
    return U2Net(config=lite, out_channel=1)


x = u2net_full()
y = x(torch.rand(1, 3, 320, 320))
# print(y)
