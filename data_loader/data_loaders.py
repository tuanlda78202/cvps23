import sys, os 
# export PYTHONPATH="${PYTHONPATH}:/cvps23"
sys.path.append(os.getcwd())

from torch.utils.data import DataLoader
from torchvision import transforms
from base.base_data_loader import BaseDataLoader
from data_loader.datasets import KNC_Dataset, Rescale, RandomCrop, ToTensorLab
from utils import mask_image_list

# Korean Name Card Data Loader

class KNC_DataLoader(BaseDataLoader):
    def __init__(self, output_size, crop_size,
                 batch_size, shuffle, validation_split, num_workers):
        
        self.output_size = output_size
        self.crop_size = crop_size
        
        # data ~ 82k, data-demo ~ 1.6k 
        self.img_list, self.mask_list = mask_image_list(dir="data")
        self.dataset = KNC_Dataset(self.img_list, self.mask_list,
                                   transform=transforms.Compose([Rescale(self.output_size),
                                                                 RandomCrop(self.crop_size),
                                                                 ToTensorLab(flag=0)]))
    
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)