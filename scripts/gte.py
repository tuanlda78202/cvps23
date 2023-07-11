# GT Encoder
import sys, os

sys.path.append(os.getcwd())
from tqdm import tqdm

import time
import numpy as np
from skimage import io
import time
import argparse
from configs.parse_config import ConfigParser

import src.dataloader.data_loaders as module_data
import src.model as module_arch

import torch
import gc
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

from src.model import ISNetGTEncoder
from src.metrics.metric import f1_mae_torch

device = "cuda" if torch.cuda.is_available() else "cpu"


def train_gte(
    train_dataloaders,
    valid_dataloaders,
    settings,
):
    config = settings["gte"]

    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config["seed"])

    print("Define GT Encoder ...")
    net = ISNetGTEncoder()

    # Resume the GT Encoder
    if config["gt_encoder_model"] != "":
        model_path = config["model_path"] + "/" + config["gt_encoder_model"]
        if torch.cuda.is_available():
            net.load_state_dict(torch.load(model_path))
            net.cuda()
        else:
            net.load_state_dict(torch.load(model_path, map_location="cpu"))
        print("gt encoder restored from the saved weights ...")
        return net  ############

    if torch.cuda.is_available():
        net.cuda()

    print("--- Define optimizer for GT Encoder---")
    optimizer = optim.Adam(
        net.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=0
    )

    model_path = config["model_path"]
    model_save_fre = config["model_save_fre"]
    max_ite = config["max_ite"]
    batch_size_train = config["batch_size_train"]
    batch_size_valid = config["batch_size_valid"]

    if not os.path.exists(model_path):
        os.mkdir(model_path)

    ite_num = config["start_ite"]  # count the total iteration number
    ite_num4val = 0  #
    running_loss = 0.0  # count the toal loss
    running_tar_loss = 0.0  # count the target output loss
    last_f1 = [0 for x in range(len(valid_dataloaders))]

    net.train()

    start_last = time.time()
    epoch_num = config["max_epoch_num"]
    notgood_cnt = 0

    for epoch in range(epoch_num):  ## set the epoch num as 100000
        tqdm_batch = tqdm(
            iterable=train_dataloaders,
            desc="Epoch {}".format(epoch),
            total=len(train_dataloaders),
            unit="it",
        )

        for i, data in enumerate(tqdm_batch):
            if ite_num >= max_ite:
                print("Training Reached the Maximal Iteration Number ", max_ite)
                exit()

            # start_read = time.time()
            ite_num = ite_num + 1
            ite_num4val = ite_num4val + 1

            # get the inputs
            labels = data["mask"]

            labels = labels.type(torch.FloatTensor)

            # wrap them in Variable
            if torch.cuda.is_available():
                labels_v = Variable(labels.cuda(), requires_grad=False)
            else:
                labels_v = Variable(labels, requires_grad=False)

            # print("time lapse for data preparation: ", time.time()-start_read, ' s')

            # y zero the parameter gradients
            # start_inf_loss_back = time.time()
            optimizer.zero_grad()

            ds, fs = net(labels_v)  # net(inputs_v)
            loss2, loss = net.compute_loss(ds, labels_v)

            loss.backward()
            optimizer.step()

            tqdm_batch.set_postfix(loss=loss.item(), tar_loss=loss2.item())

            # del outputs, loss
            del ds, loss2, loss
            # end_inf_loss_back = time.time() - start_inf_loss_back

            """
            print(
                "GT Encoder Training>>>"
                + model_path.split("/")[-1]
                + " - [epoch: %3d/%3d, ite: %d] train loss: %3f, tar: %3f"
                % (
                    epoch + 1,
                    epoch_num,
                    ite_num,
                    running_loss / ite_num4val,
                    running_tar_loss / ite_num4val,
                )
            )
            """

        model_name = "/GTENCODER-gpu_e_" + str(epoch) + ".pth"

        torch.save(net.state_dict(), model_path + model_name)

        print("saving ckpt gte")


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Salient Object Detection")

    args.add_argument(
        "-d",
        "--device",
        default="cuda",
        type=str,
        help="indices of GPUs to enable (default: all)",
    )

    args.add_argument(
        "-c",
        "--config",
        default="configs/dis/isnetdis_scratch_1xb4-1k_knc-1024x1024.yaml",
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

    config = ConfigParser.from_args(args)

    # Data
    data_loader = config.init_obj("data_loader", module_data)
    valid_data_loader = data_loader.split_validation()

    # Model
    model = config.init_obj("arch", module_arch)

    # Settings YAML load config
    train_gte(
        train_dataloaders=data_loader,
        valid_dataloaders=valid_data_loader,
        settings=config,
    )
