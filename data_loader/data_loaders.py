from torchvision import datasets, transforms
from base import BaseDataLoader
from datasets import KNC


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

# Korean Name Card Data Loader
class KNCDataLoader(BaseDataLoader):
    def __init__(self, dataset, batch_size, img_size = 512, shuffle=True, train_portion=1.0, num_workers=1, pin_memory=False, drop_last=False, training=True):
        transforms_list = [transforms.Resize((int(img_size), int(img_size)),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))]
        tranf = transforms.Compose(transforms_list)
        
        self.data_dir = data_dir
        self.dataset = KNC(self.data_dir, transform=transf)
        super().__init__(self.dataset, batch_size, shuffle, train_portion, num_workers, pin_memory=pin_memory, drop_last=drop_last)
    