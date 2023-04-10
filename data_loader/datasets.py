import glob
import os 
import numpy as np 
from PIL import Image
from torch.utils.data import Dataset

# Korean Name Card Datasets
class KNC(Dataset):
    def __init__(self, data_dir, transform):
        self.data_dir = data_dir
        self.transform = transform
        self.img_paths = glob.glob(os.path.join(self.data_dir, "*.jpg"))
        self.len = len(self.img_paths)
        self.img_paths = self.img_paths[:self.len]

    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])
        
        if self.transform:
            img = self.transform(img)
            
        return (img,1)