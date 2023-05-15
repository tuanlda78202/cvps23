from __future__ import print_function, division
import random
from skimage import io
import torch.nn.functional as F
import numpy as np
import os
import glob
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import normalize
from collections import defaultdict


class KNCDataset(Dataset):

    """Korean Name Card Datasets"""

    def __init__(self, img_list, mask_list, transform):
        self.img_list = img_list
        self.mask_list = mask_list
        self.len = len(self.img_list)
        self.transform = transform

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        # H x W x 3
        img = io.imread(self.img_list[idx])
        img_idx = np.array([idx])

        # Iter mask list
        if len(self.mask_list) == 0:
            mask_rbg = np.zeros(img.shape)
        else:
            mask_rbg = io.imread(self.mask_list[idx])

        mask = np.zeros(mask_rbg.shape[0:2])

        # Check Mask 3 channels
        if len(mask_rbg.shape) == 3:
            mask = mask_rbg[:, :, 0]
        elif len(mask_rbg.shape) == 2:
            mask = mask_rbg

        # Assure img and mask has 3 channels
        if len(img.shape) == 3 and len(mask.shape) == 2:
            mask = mask[:, :, np.newaxis]

        elif len(img.shape) == 2 and len(mask.shape) == 2:
            img = img[:, :, np.newaxis]
            mask = mask[:, :, np.newaxis]

        sample = {"img_idx": img_idx, "img": img, "mask": mask}

        # Transform
        if self.transform:
            sample = self.transform(sample)

        return sample


class Rescale(object):
    """Rescale size with output quality defined"""

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        # Image index, Image and Mask
        img_idx, img, mask = sample["img_idx"], sample["img"], sample["mask"]

        img = torch.squeeze(
            F.interpolate(
                torch.unsqueeze(img, dim=0), self.output_size, mode="bilinear"
            ),
            dim=0,
        )

        mask = torch.squeeze(
            F.interpolate(
                torch.unsqueeze(mask, dim=0), self.output_size, mode="bilinear"
            ),
            dim=0,
        )

        return {"img_idx": img_idx, "img": img, "mask": mask}


class RandomCrop(object):
    """Data Augmentation random 0.5"""

    def __init__(self, output_size, prob=0.5):
        self.prob = prob
        assert isinstance(output_size, (int, tuple))

        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        # Image index, Image and Mask
        img_idx, img, mask = sample["img_idx"], sample["img"], sample["mask"]

        # Reverse image (array) [::-1]
        if random.random() >= self.prob:
            img, mask = img[::-1], mask[::-1]

        # numpy: H x W x 3
        if img.shape[0] > img.shape[2]:
            h, w = img.shape[:2]

        # torch: 3 x H x W
        elif img.shape[0] < img.shape[2]:
            h, w = img.shape[1:]

        new_h, new_w = self.output_size

        # Random range
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        # Crop
        img = img[top : top + new_h, left : left + new_w]
        mask = mask[top : top + new_h, left : left + new_w]

        return {"img_idx": img_idx, "img": img, "mask": mask}


class RandomHFlip(object):
    """Random Horizontal Flip image"""

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        img_idx, img, mask = sample["img_idx"], sample["img"], sample["mask"]

        # 3 x H x W
        if random.random() >= self.prob:
            img = torch.flip(img, dims=[2])
            mask = torch.flip(mask, dims=[2])

        return {"img_idx": img_idx, "img": img, "mask": mask}


class RandomVFlip(object):
    """Random Vertical Flip image"""

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        img_idx, img, mask = sample["img_idx"], sample["img"], sample["mask"]

        # 3 x H x W
        if random.random() >= self.prob:
            img = torch.flip(img, dims=[1])
            mask = torch.flip(mask, dims=[1])

        return {"img_idx": img_idx, "img": img, "mask": mask}


class Normalize(object):
    """Image Normalization
    output[channel] = (input[channel] - mean[channel]) / std[channel]
    """

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img_idx, img, mask = sample["img_idx"], sample["img"], sample["mask"]

        img = normalize(img, self.mean, self.std)

        return {"img_idx": img_idx, "img": img, "mask": mask}


class ImageProcess:
    def __init__(self, dir=None, img_path=None, img=None, gt=None, size=None):
        self.dir = dir
        self.img_path = img_path
        self.img = img
        self.gt = gt
        self.size = size

    def mask_image_list(self):
        data_dir = os.path.join(os.getcwd(), self.dir + os.sep)
        img_dir = os.path.join("img" + os.sep)
        mask_dir = os.path.join("mask" + os.sep)

        img_ext, mask_ext = ".jpg", ".png"

        img_list = glob.glob(data_dir + img_dir + "*" + img_ext)
        mask_list = []

        for img_path in img_list:
            full_name = img_path.split(os.sep)[-1]

            name_ext = full_name.split(".")
            name_list = name_ext[0:-1]
            img_idx = name_list[0]

            for i in range(1, len(name_list)):
                img_idx = img_idx + "." + name_list[i]

            mask_list.append(data_dir + mask_dir + img_idx + mask_ext)

        return img_list, mask_list

    def img_reader(self):
        """Load image from data folder"""

        return io.imread(self.img_path)

    def preprocess_img(self, img, size):
        """Load image, read to np and convert to torch"""

        # img: H x W x 3
        img, size = self.img, self.size

        # Make sure shape of images has 3 channels
        if len(img.shape) < 3:
            img = img[:, :, np.newaxis]
        # If binary
        elif img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)

        # np: H x W x 3
        img_tensor = torch.tensor(img.copy(), dtype=torch.float32)
        # torch: 3 x H x W
        img_tensor = torch.transpose(torch.transpose(img_tensor, 1, 2), 0, 1)

        # Size input model
        if len(size) < 2:
            return img_tensor, img.shape[:2]

        else:
            # 1 x 3 x H x W (bcs input interpolate need form "batch")
            img_tensor = torch.unsqueeze(img_tensor, dim=0)
            # 1 x 3 x size x size
            img_tensor = F.interpolate(img_tensor, size, mode="bilinear")
            # 3 x H x W
            img_tensor = torch.squeeze(img_tensor, dim=0)

        return img_tensor.type(torch.uint8), img.shape[:2]

    def preprocess_gt(self, gt, size):
        """Load GT, read to np and convert to torch"""

        gt, size = self.gt, self.size

        # H x W x 3 > just need H x W x 1
        if len(gt.shape) > 2:
            gt = gt[:, :, 0]

        # 1 x H x W x 1
        gt_tensor = torch.unsqueeze(torch.tensor(gt, dtype=torch.uint8), dim=0)

        if len(size) < 2:
            return gt_tensor.type(torch.uint8), gt.shape[:2]

        else:
            gt_tensor = torch.unsqueeze(
                torch.tensor(gt_tensor, dtype=torch.float32), dim=0
            )
            gt_tensor = F.interpolate(gt_tensor, size, mode="bilinear")
            gt_tensor = torch.squeeze(gt_tensor, dim=0)

        return gt_tensor.type(torch.uint8), gt.shape[:2]
