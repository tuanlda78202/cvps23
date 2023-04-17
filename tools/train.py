import argparse
import collections
import torch
import numpy as np
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.architecture as module_arch
from configs.parse_config import ConfigParser
from trainer import Trainer
import warnings
import sys
import os 

# export PYTHONPATH="${PYTHONPATH}:/cvps23"
warnings.filterwarnings("ignore")

# fix random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config):
    # Logging
    logger = config.get_logger('train')

    # Data Loader 
    data_loader = config.init_obj('data_loader', module_data)
    valid_data_loader = data_loader.split_validation()
    
    # Model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # Device GPU training
    device = "mps" if torch.backends.mps.is_available() else ("cuda:0" if torch.cuda.is_available() else "cpu")
    # CPU 
    device = "cpu"
    model = model.to(device)

    # Loss & Metrics
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]
    
    
    # Optimizer & LR scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    # Loop 
    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      device=device,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler)

    trainer.train()


if __name__ == '__main__':
    
    args = argparse.ArgumentParser(description='Salient Object Detection')
    args.add_argument('-c', '--config', default="configs/u2net/u2net-lite_scratch_1xb8-1k_knc-320x320.yaml", type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default="mps", type=str,
                      help='indices of GPUs to enable (default: all)')

    # Custom CLI options to modify configuration from default values given in yaml file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size'),
        CustomArgs(["--ep", "--epochs"], type=int, target="trainer;epochs")
    ]
    
    config = ConfigParser.from_args(args, options)
    
    main(config)
