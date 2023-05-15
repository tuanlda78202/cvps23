import argparse
from configs.parse_config import ConfigParser
import model.architecture as module_arch
import os
import numpy as np
from skimage import io
from glob import glob
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import normalize
import collections


def main(config):
    # Inference dataset
    dataset_path = config["inference"]["dataset_path"]
    model_path = config["inference"]["model_path"]
    result_path = config["inference"]["result_path"]
    input_size = config["inference"]["input_size"]

    im_list = glob(dataset_path + "/*.jpg") + glob(dataset_path + "/*.png")

    model = config.init_obj("arch", module_arch)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    if device == "cuda:0":
        model.load_state_dict(torch.load(model_path))
        model = model.to(device)
    else:
        model.load_state_dict(torch.load(model_path, map_location="cpu"))

    model.eval()

    with torch.no_grad():
        for idx, img_path in tqdm(enumerate(im_list), total=len(im_list)):
            img = io.imread(img_path)

            img_tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
            img_tensor = F.upsample(
                torch.unsqueeze(img_tensor, 0), input_size, mode="bilinear"
            ).type(torch.uint8)

            image = torch.divide(img_tensor, 255.0)
            image = normalize(image, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])

            if torch.cuda.is_available():
                image = image.cuda()

            result = model(image)
            result = torch.squeeze(
                F.interpolate(result[0][0], img_tensor.shape[:2], mode="bilinear"),
                dim=0,
            )

            result = (result - torch.max(result)) / (
                torch.max(result) - torch.min(result)
            )

            img_name = img_path.split("/")[-1].split(".")[0]

            io.imsave(
                os.path.join(result_path, img_name + ".png"),
                (result * 255).permute(1, 2, 0).cpu().data.numpy().astype(np.uint8),
            )


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Inference SOD")

    args.add_argument(
        "-c",
        "--config",
        default="configs/dis/isnetdis_scratch_1xb8-1k_knc-1024x1024.yaml",
        type=str,
        help="config file path (default: None)",
    )

    args.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: None)",
    )

    args.add_argument(
        "-d",
        "--device",
        default="mps",
        type=str,
        help="indices of GPUs to enable (default: all)",
    )

    # Custom CLI options to modify configuration from default values given in yaml file.
    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")

    options = [
        CustomArgs(
            ["--bs", "--batch_size"], type=int, target="data_loader;args;batch_size"
        ),
        CustomArgs(["--ep", "--epochs"], type=int, target="trainer;epochs"),
    ]

    config = ConfigParser.from_args(args, options)

    main(config)
