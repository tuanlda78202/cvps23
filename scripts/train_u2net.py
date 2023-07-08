import sys, os

sys.path.append(os.getcwd())

import argparse
import collections
import torch
import numpy as np
import src.dataloader.data_loaders as module_data
import src.metrics.loss as module_loss
import src.metrics.metric as module_metric
import src.model as module_arch
from configs.parse_config import ConfigParser
from src.trainer import U2NetTrainer

import warnings

warnings.filterwarnings("ignore")

SEED = 42
torch.manual_seed(SEED)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
np.random.seed(SEED)


def main(config):
    # Logging
    logger = config.get_logger("train")

    # Data Loader
    data_loader = config.init_obj("data_loader", module_data)
    valid_data_loader = data_loader.split_validation()

    # Model architecture
    model = config.init_obj("arch", module_arch)
    logger.info(model)

    # Device GPU training
    device = torch.device("cuda")
    model = model.to(device)

    # Loss & Metrics
    criterion = getattr(module_loss, config["loss"])
    metrics = [getattr(module_metric, met) for met in config["metrics"]]

    # Optimizer & LR scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj("optimizer", torch.optim, trainable_params)
    lr_scheduler = config.init_obj("lr_scheduler", torch.optim.lr_scheduler, optimizer)

    # U2NetTrainer
    trainer = U2NetTrainer(
        model,
        criterion,
        metrics,
        optimizer,
        config=config,
        device=device,
        data_loader=data_loader,
        valid_data_loader=valid_data_loader,
        lr_scheduler=lr_scheduler,
    )

    trainer.train()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Salient Object Detection")

    args.add_argument(
        "-c",
        "--config",
        default="configs/u2net/u2net-lite_scratch_1xb16-1k_knc-320x320.yaml",
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
        default="cuda",
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
