from __future__ import print_function, division

import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import albumentations


class KNC_Albumentation_Dataset(Dataset):
    """
    A custom Dataset class must implement three functions: __init__, __len__, and __getitem__.
    Dataset retrieves dataset’s features and labels ONE sample at a time
    """

    def __init__(self, img_list, mask_list, transform):
        self.img_list = img_list
        self.mask_list = mask_list
        self.len = len(self.img_list)
        self.transform = transform

    def __len__(self):
        return self.len

    def __getitem__(self, idx):

        """
        The __getitem__ function loads and returns a sample from the dataset at the given index idx.
        Based on the index, it identifies the image’s location on disk, converts that to a tensor using read_image,
        retrieves the corresponding label from the csv data in self.img_labels, calls the transform functions on them (if applicable),
        and returns the tensor image and corresponding label in a tuple.
        """
        img_idx = np.array([idx])

        img = cv2.imread(self.img_list[idx])

        # Iter mask list
        if len(self.mask_list) == 0:
            mask_rbg = np.zeros(img.shape)
        else:
            mask_rbg = cv2.imread(self.mask_list[idx])

        mask = np.zeros(mask_rbg.shape[0:2])

        # Check Mask 3 channels
        if len(mask_rbg.shape) == 3:
            mask = mask_rbg[:, :, 0]
        elif len(mask_rbg.shape) == 2:
            mask = mask_rbg

        # Assure img & mask has 3 channels
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

