import gc

import torch
import numpy as np
import cv2
import os
from torchvision.utils import make_grid


def l2_regularisation(m):
    l2_reg = None

    for W in m.parameters():
        if l2_reg is None:
            l2_reg = W.norm(2)
        else:
            l2_reg = l2_reg + W.norm(2)
    return l2_reg


def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=5):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay


# linear annealing to avoid posterior collapse
def linear_annealing(init, fin, step, annealing_steps):
    """Linear annealing of a parameter."""
    if annealing_steps == 0:
        return fin
    assert fin > init
    delta = fin - init
    annealed = min(init + delta * step / annealing_steps, fin)

    return annealed


def visualize_pred_prior(pred, wandb):
    # save_path = './image_results/prior/'
    # if not os.path.exists(save_path):
    #     os.mkdir(save_path)
    images = wandb.Image(make_grid(pred[:32], nrow=8))
    wandb.log({"prior": images})
    del images
    gc.collect()
    # for kk in range(pred.shape[0]):
    #     pred_edge_kk = pred[kk,:,:,:]
    #     pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
    #     pred_edge_kk *= 255.0
    #     pred_edge_kk = pred_edge_kk.astype(np.uint8)
    #
    #     name = '{:02d}_prior.png'.format(kk)
    #     cv2.imwrite(save_path + name, pred_edge_kk)


def visualize_pred_post(pred, wandb):
    # save_path = './image_results/post/'
    # if not os.path.exists(save_path):
    #     os.mkdir(save_path)

    images = wandb.Image(make_grid(pred[:32], nrow=8))
    wandb.log({"posterior": images})
    del images
    gc.collect()
    # for kk in range(pred.shape[0]):
    #     pred_edge_kk = pred[kk,:,:,:]
    #     pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
    #     pred_edge_kk *= 255.0
    #     pred_edge_kk = pred_edge_kk.astype(np.uint8)
    #
    #     name = '{:02d}_post.png'.format(kk)
    #     cv2.imwrite(save_path + name, pred_edge_kk)


def visualize_gt(var_map, wandb):
    # save_path = './image_results/ground_truth/'
    # if not os.path.exists(save_path):
    #     os.mkdir(save_path)

    images = wandb.Image(make_grid(var_map[:32], nrow=8))
    wandb.log({"ground truth": images})
    del images
    gc.collect()
    # for kk in range(var_map.shape[0]):
    #     pred_edge_kk = var_map[kk,:,:,:]
    #     pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
    #     pred_edge_kk *= 255.0
    #     pred_edge_kk = pred_edge_kk.astype(np.uint8)
    #
    #     name = '{:02d}_gt.png'.format(kk)
    #     cv2.imwrite(save_path + name, pred_edge_kk)


class AvgMeter(object):
    def __init__(self, num=40):
        self.num = num
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.losses = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.losses.append(val)

    def show(self):
        a = len(self.losses)
        b = np.maximum(a-self.num, 0)
        c = self.losses[b:]

        return torch.mean(torch.stack(c))