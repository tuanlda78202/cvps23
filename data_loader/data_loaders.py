from torch.utils.data import DataLoader
from torchvision import transforms
from base.base_data_loader import BaseDataLoader
from data_loader.datasets import KNC_Dataset, Rescale, RandomCrop, ToTensorLab

# Korean Name Card Data Loader

class KNC_DataLoader(BaseDataLoader):
    def __init__(self, img_list, mask_list, output_size, dataset):
        super().__init__()
        self.img_list = img_list
        self.mask_list = mask_list
        self.output_size = output_size
        
        self.dataset = KNC_Dataset(self.img_list, self.mask_list,
                                   transform=transforms.Compose([Rescale(self.output_size),
                                                                 RandomCrop(self.output_size),
                                                                 ToTensorLab(flag=0)]))