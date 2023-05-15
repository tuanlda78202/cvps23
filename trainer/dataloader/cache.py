# Data Cache for HighRes data of DIS
from __future__ import print_function, division

import numpy as np
from copy import deepcopy
import json
from tqdm import tqdm
import os
from trainer.dataloader.datasets import ImageProcess

import torch
from collections import defaultdict
from trainer.dataloader.datasets import KNCDataset


class KNCCache(KNCDataset):
    """Data cache all the images and gts into a single pytorch tensor"""

    def __init__(
        self,
        img_list,
        mask_list,
        transform,
        data_info,
        cache_size=[],
        cache_path="./cache",
        cache_file_name="dataset.json",
        cache_boost=False,
    ):
        super().__init__(img_list, mask_list, transform)

        self.cache_size = cache_size
        self.cache_path = cache_path

        # Cache numpy as well regardless of cache_boost
        self.cache_file_name = cache_file_name
        self.cache_boost = cache_boost
        if self.cache_boost:
            self.cache_boost_name = cache_file_name.split(".json")[0]

        self.imgs_pt = None
        self.gts_pt = None

        # Dataset
        self.dataset = defaultdict()
        self.dataset["data_name"] = data_info["dataset_name"]
        self.dataset["img_path"] = data_info["img_path"]
        self.dataset["gt_path"] = data_info["gt_path"]
        self.dataset["img_ext"] = data_info["img_ext"]
        self.dataset["gt_ext"] = data_info["gt_ext"]
        self.dataset["img_name"] = [
            img_path.split(os.sep)[-1].split(".") for img_path in data_info["img_path"]
        ]

        self.dataset = self.manage_cache(self.dataset["data_name"])

    def __getitem__(self, idx):
        if self.cache_boost and self.imgs_pt is not None:
            img = self.imgs_pt[idx]
            gt = self.gts_pt[idx]

        else:
            img_pt_path = os.path.join(
                self.cache_path,
                os.sep.join(self.dataset["img_path"][idx].split(os.sep)[-2:]),
            )

            gt_pt_path = os.path.join(
                self.cache_path,
                os.sep.join(self.dataset["gt_path"][idx].split(os.sep)[-2:]),
            )

            img = torch.load(img_pt_path)
            gt = torch.load(gt_pt_path)

        img = torch.divide(img, 255.0)
        gt = torch.divide(gt, 255.0)

        sample = {"img_idx": torch.from_numpy(np.array(idx)), "img": img, "mask": gt}

        # Transform
        if self.transform:
            sample = self.transform(sample)

        return sample

    def manage_cache(self, dataset_name):
        """Create folder cache, check if not cache files are there, then cache"""

        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)

        cache_folder = os.path.join(
            self.cache_path,
            "_" + dataset_name + "_" + "x".join([str(x) for x in self.cache_size]),
        )

        if not os.path.exists(cache_folder):
            return self.cache(cache_folder)

        return self.load_cache(cache_folder)

    def load_cache(self, cache_folder):
        """Load Torch tensor into RAM"""
        with open(os.path.join(cache_folder, self.cache_file_name), "r") as json_file:
            cache_data = json.load(json_file)

        # If self.cache_boost true, load imgs and GT numpy into the RAM
        # otw tensor will be loaded
        if self.cache_boost:
            self.imgs_pt = torch.load(cache_data["imgs_pt_dir"], map_location="cpu")
            self.gts_pt = torch.load(cache_data["gts_pt_dir"], map_location="cpu")

        return cache_data

    def cache(self, cache_folder):
        os.mkdir(cache_folder)
        cached_dataset = deepcopy(self.dataset)

        image_process = ImageProcess()
        pt_dict = defaultdict(list)

        for idx, img_path in tqdm(
            enumerate(self.dataset["img_path"]), total=len(self.dataset["img_path"])
        ):
            img_id = cached_dataset["img_name"][idx]

            img = image_process.img_reader(img_path)
            gt = image_process.img_reader(self.dataset["gt_path"][idx])

            img = image_process.preprocess_img(img, self.cache_size)
            gt = image_process.preprocess_gt(gt, self.cache_size)

            # Cache
            img_cache_file = os.path.join(
                cache_folder, self.dataset["data_name"] + "_" + img_id + "_img.pt"
            )

            gt_cache_file = os.path.join(
                cache_folder, self.dataset["data_name"] + "_" + img_id + "_gt.pt"
            )

            # Save
            torch.save(img, img_cache_file)
            torch.save(gt, gt_cache_file)

            cached_dataset["img_path"][idx] = img_cache_file
            cached_dataset["gt_path"][idx] = gt_cache_file

            if self.cache_boost:
                pt_dict["imgs_pt_list"].append(torch.unsqueeze(img, 0))
                pt_dict["gts_pt_list"].append(torch.unsqueeze(gt, 0))


def data_dict(dataset):
    image_process = ImageProcess(dir="data_demo")
    img_list, mask_list = image_process.mask_image_list()

    data_info = {
        "dataset_name": dataset["name"],
        "img_path": img_list,
        "gt_path": mask_list,
        "img_ext": dataset["im_ext"],
        "gt_ext": dataset["gt_ext"],
        "cache_dir": dataset["cache_dir"],
    }
    return data_info
