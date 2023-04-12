import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import MaxPool2d, Conv2d
import torch.nn.functional as F
#from torchsummary import summary

from base.base_rsu import _up_same, _size_map, RSU

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
        return super().__str__() + '\nTrainable parameters: {}'.format(params)  
    
    def _make_layers(self, config):
        # Height of RSU Block
        self.height = int((len(config)+1) / 2)
        
        # MaxPood 2D 
        self.add_module("down_sample", MaxPool2d(kernel_size=2, stride=2, ceil_mode=True))
        
        # Built RSU Block & Side Layer
        for key, value in config.items():
            
            # RSU Block
            self.add_module(key, RSU(value[0], *value[1]))
            
            # Side Layer
            if value[2] > 0:
                self.add_module(f"side{value[0][-1]}", Conv2d(value[2], self.out_channel, kernel_size=3, padding=1))
        # Fuse Layer
        self.add_module("out_conv", Conv2d(int(self.height * self.out_channel), self.out_channel, kernel_size=1))
    
    def forward(self, x):
        sizes = _size_map(x, self.height)
        maps = []  # storage for maps

        # Side saliency map
        def unet(x, height=1):
            if height < 6:
                x1 = getattr(self, f'stage{height}')(x)
                
                x2 = unet(getattr(self, 'down_sample')(x1), height + 1)
                
                x = getattr(self, f'stage{height}d')(torch.cat((x2, x1), 1))
                side(x, height)
                
                return _up_same(x, sizes[height - 1]) if height > 1 else x
            else:
                x = getattr(self, f'stage{height}')(x)
                side(x, height)
                return _up_same(x, sizes[height - 1])

        def side(x, h):
            # side output saliency map (before sigmoid)
            x = getattr(self, f'side{h}')(x)
            x = _up_same(x, sizes[1])
            maps.append(x)

        def fuse():
            # fuse saliency probability maps
            maps.reverse()
            x = torch.cat(maps, 1)
            x = getattr(self, 'out_conv')(x)
            maps.insert(0, x)
            return [torch.sigmoid(x) for x in maps]

        unet(x)
        maps = fuse()
        return maps

# Config for building RSUs and sides
# {stage : [name, (height(L), in_ch, mid_ch, out_ch, dilated), side]}
def u2net_full():
    full = {
        # config for building RSUs and sides
        # {stage : [name, (height(L), in_ch, mid_ch, out_ch, dilated), side]}
        'stage1': ['En_1', (7, 3, 32, 64), -1],
        'stage2': ['En_2', (6, 64, 32, 128), -1],
        'stage3': ['En_3', (5, 128, 64, 256), -1],
        'stage4': ['En_4', (4, 256, 128, 512), -1],
        'stage5': ['En_5', (4, 512, 256, 512, True), -1],
        'stage6': ['En_6', (4, 512, 256, 512, True), 512],
        'stage5d': ['De_5', (4, 1024, 256, 512, True), 512],
        'stage4d': ['De_4', (4, 1024, 128, 256), 256],
        'stage3d': ['De_3', (5, 512, 64, 128), 128],
        'stage2d': ['De_2', (6, 256, 32, 64), 64],
        'stage1d': ['De_1', (7, 128, 16, 64), 64],
    }
    return U2Net(config=full, out_channel=1)    

def u2net_lite():
    lite = {
        # co for building RSUs and sides
        # {stage : [name, (height(L), in_ch, mid_ch, out_ch, dilated), side]}
        'stage1': ['En_1', (7, 3, 16, 64), -1],
        'stage2': ['En_2', (6, 64, 16, 64), -1],
        'stage3': ['En_3', (5, 64, 16, 64), -1],
        'stage4': ['En_4', (4, 64, 16, 64), -1],
        'stage5': ['En_5', (4, 64, 16, 64, True), -1],
        'stage6': ['En_6', (4, 64, 16, 64, True), 64],
        'stage5d': ['De_5', (4, 128, 16, 64, True), 64],
        'stage4d': ['De_4', (4, 128, 16, 64), 64],
        'stage3d': ['De_3', (5, 128, 16, 64), 64],
        'stage2d': ['De_2', (6, 128, 16, 64), 64],
        'stage1d': ['De_1', (7, 128, 16, 64), 64],
    }
    return U2Net(config=lite, out_channel=1)    

# Model parameters
# print(summary(u2net_full(), input_size = (3, 320, 320)))