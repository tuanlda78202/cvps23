import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import MaxPool2d, Upsample
import torch.nn.functional as F


def _up_same(x, size):
    return nn.Upsample(size=size, mode="bilinear", align_corners=False)(x)


def _size_map(x, height):
    """
    {height: size} for Up-sample
    """
    sizes = {}
    size = list(x.shape[-2:])

    for h in range(1, height):
        sizes[h] = size
        size = [math.ceil(w / 2) for w in size]

    return sizes


# RELu + BatchNorm + DPConv
class RBDP(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, dilate=1, stride=1):
        super(RBDP, self).__init__()
        # kernel 1X1
        self.conv1x1 = nn.Conv2d(in_ch, out_ch, 1)
        # global average pooling
        self.GAP = nn.AdaptiveAvgPool2d((1, 1))
        # MLP
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.linear = nn.Linear(out_ch, 3)
        self.softmax = nn.Softmax(dim=1)

        # Paramidal convolution
        self.conv3x3 = nn.Conv2d(out_ch, out_ch, 3, padding=1 * dilate, dilation=1 * dilate, stride=stride)
        self.conv5x5 = nn.Conv2d(out_ch, out_ch, 5, padding=2 * dilate, dilation=1 * dilate, stride=stride)
        self.conv7x7 = nn.Conv2d(out_ch, out_ch, 7, padding=3 * dilate, dilation=1 * dilate, stride=stride)

        # kernel 1X1 final
        self.conv1x1_final = nn.Conv2d(3 * out_ch, out_ch, 1)

        # ReLU + BatchNorm
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1x1(x)
        hx = x
        # GAP + MLP
        hx = self.GAP(hx)
        hx = self.flatten(hx)
        hx = self.relu(hx)
        hx = self.linear(hx)
        hx = self.softmax(hx)
        # Paramidal convolution
        out1 = self.conv3x3(x)
        out2 = self.conv5x5(x)
        out3 = self.conv7x7(x)

        # channel wise
        out1 = out1 * hx[:, 0].view(-1, 1, 1, 1)
        out2 = out2 * hx[:, 1].view(-1, 1, 1, 1)
        out3 = out3 * hx[:, 2].view(-1, 1, 1, 1)
        
        # concatenation
        out_fuse = torch.cat((out1, out2, out3), 1)
        # conv1x1_final
        y = self.conv1x1_final(out_fuse)

        return self.relu_s1(self.bn_s1(y + x))
# RELu + BatchNorm + Conv
class RBC(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, dilate=1, stride=None, padding=None):
        super().__init__()
        self.conv = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=3,
            stride=1,
            padding=1 * dilate,
            dilation=1 * dilate,
        )
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        return x

# Residual U-blocks
class RSU(nn.Module):
    def __init__(
        self, name, height, in_channel, mid_channel, out_channel, dilated=False
    ):
        super().__init__()
        self.name = name
        self.height = height
        self.dilated = dilated
        self._make_layers(height, in_channel, mid_channel, out_channel, dilated)

    def _make_layers(self, height, in_channel, mid_channel, out_channel, dilated=False):
        if height == 7:
            # RELu + BatchNorm + Convolution Input & MaxPool 2D
            self.add_module("rbc_in", RBDP(in_channel, out_channel))
            self.add_module(
                "down_sample", nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
            )

            # Encoder & Decoder RSU
            self.add_module(f"rbc1", RBDP(out_channel, mid_channel))
            self.add_module(f"rbc1d", RBDP(mid_channel * 2, out_channel))

            # Length of RSU blocks = 7
            for i in range(2, height):
                # Dilated Kernel
                dilate = 1 if not dilated else 2 ** (i - 1)

                self.add_module(f"rbc{i}", RBDP(mid_channel, mid_channel, dilate=dilate))
                self.add_module(
                    f"rbc{i}d", RBDP(mid_channel * 2, mid_channel, dilate=dilate)
                )

            dilate = 2 if not dilated else 2 ** (height - 1)
            self.add_module(f"rbc{height}", RBDP(mid_channel, mid_channel, dilate=dilate))
        else:
            # RELu + BatchNorm + Convolution Input & MaxPool 2D
            self.add_module("rbc_in", RBC(in_channel, out_channel))
            self.add_module(
                "down_sample", nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
            )

            # Encoder & Decoder RSU
            self.add_module(f"rbc1", RBC(out_channel, mid_channel))
            self.add_module(f"rbc1d", RBC(mid_channel * 2, out_channel))

            # Length of RSU blocks = 7
            for i in range(2, height):
                # Dilated Kernel
                dilate = 1 if not dilated else 2 ** (i - 1)

                self.add_module(f"rbc{i}", RBC(mid_channel, mid_channel, dilate=dilate))
                self.add_module(
                    f"rbc{i}d", RBC(mid_channel * 2, mid_channel, dilate=dilate)
                )

            dilate = 2 if not dilated else 2 ** (height - 1)
            self.add_module(f"rbc{height}", RBC(mid_channel, mid_channel, dilate=dilate))

    def forward(self, x):
        # Input
        sizes = _size_map(x, self.height)
        x = self.rbc_in(x)

        # U-Net like symmetric encoder-decoder structure
        def unet(x, height=1):
            if height < self.height:
                x1 = getattr(self, f"rbc{height}")(x)

                # Python recursion
                if not self.dilated and height < self.height - 1:
                    x2 = unet(getattr(self, "down_sample")(x1), height + 1)
                else:
                    x2 = unet(x1, height + 1)

                # Concatenation
                x = getattr(self, f"rbc{height}d")(torch.cat((x2, x1), 1))

                return (
                    _up_same(x, sizes[height - 1])
                    if (not self.dilated and height > 1)
                    else x
                )
            else:
                return getattr(self, f"rbc{height}")(x)

        # Addition
        return x + unet(x)
