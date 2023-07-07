from src.dataloader.datasets import KNCDataset
from src.dataloader.datasets import *
from src.dataloader.datasets import ImageProcess
from torchvision import transforms

image_process = ImageProcess(dir="data_demo")
img_list, mask_list = image_process.mask_image_list()
data = KNCDataset(
    img_list,
    mask_list,
    transform=transforms.Compose(
        [
            Rescale(320),
            RandomCrop(288),
            Normalize(),
        ]
    ),
)

print(data[0]["img"].shape)
