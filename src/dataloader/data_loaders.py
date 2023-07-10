from torchvision import transforms
from src.base.base_data_loader import BaseDataLoader
from src.dataloader.datasets import *
from src.dataloader.datasets import ImageProcess
import sys
import os

sys.path.append(os.getcwd())


class KNCDataLoader(BaseDataLoader):
    """Korean Name Card Data Loader (data ~ 82k, data-demo ~ 1.6k)"""

    def __init__(
        self, output_size, crop_size, batch_size, shuffle, validation_split, num_workers
    ):
        self.output_size = output_size
        self.crop_size = crop_size

        image_process = ImageProcess(dir="../sod_data")
        self.img_list, self.mask_list = image_process.mask_image_list()

        self.dataset = KNCDataset(
            self.img_list,
            self.mask_list,
            transform=transforms.Compose(
                [
                    Rescale(self.output_size),
                    RandomCrop(self.crop_size),
                    NormTensor(),
                ]
            ),
        )

        super().__init__(
            self.dataset, batch_size, shuffle, validation_split, num_workers
        )
