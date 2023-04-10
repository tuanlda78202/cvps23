# Salient Object Detection for Korean Namecard

This is the source code for the project "Salient Object Detection for Korean Namecard" of the course "Computer Vision" Summer 2023.

In this project, we will investigate the effect of data augmentation in training GANs
by experimenting it over a various methods to see whether and how it works.

<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 ordere  dList=false} -->

<!-- code_chunk_output -->

- [Salient Object Detection for Korean Namecard](#salient-object-detection-for-korean-namecard)
  - [Project Structure](#project-structure)
  - [Usage](#usage)
    - [Config file format](#config-file-format)
    - [Using config files](#using-config-files)
    - [Resuming from checkpoints](#resuming-from-checkpoints)
    - [Evaluating](#evaluating)
  - [Acknowledgements](#acknowledgements)

<!-- /code_chunk_output -->

## Project Structure

```
cvp/
│
├── train.py - main script to start training
├── eval.py - script to compute FID score on each saved checkpoint of a specified model
│
├── parse_config.py - class to handle config file and cli options
│
├── augment/ - implementation of data augmentation methods
│
├── base/ - abstract base classes
│   ├── base_data_loader.py
│   ├── base_gan.py
│   └── base_gan_trainer.py
│
├── configs/ - all training config files go here
│
├── data_loader/ - anything about data loading goes here
│   ├── data_loaders.py
│   └── custom_datasets.py
│
├── data/ - default directory for storing input data
│
├── model/
│   ├── models/ - all model architectures go here
│   ├── modules.py - custom modules
│   └── loss.py - loss functions
│
├── saved/
│   ├── models/ - trained models are saved here
│   └── log/ - default logdir for tensorboard and logging output
│
├── trainer/ - all trainer classes go here
│   └── ...
│
├── logger/ - module for tensorboard visualization and logging
│   ├── visualization.py
│   ├── logger.py
│   └── logger_config.json
│
├── utils/ - utility functions
│   ├── utils.py
│   └── ...
│
...
```

## Usage

Install the required packages:

```
pip install -r requirements.txt
```

Choose a config file from `configs/` and run:

```
python train.py --config configs/<config_file>
```

### Config file format

Config files are in `.json` format:

```javascript
{
  "name": "Cifar10_AUG_GAN",        // training session name
  "n_gpu": 1,                   // number of GPUs to use for training.

  "arch": {
    "type": "SNGAN32",       // name of model architecture to train
    "args": {
        "latent_dim": 128,   // latent dimension
    }
  },
  "data_loader": {
    "type": "Cifar10DataLoader",       // selecting data loader
    "args":{
      "data_dir": "data/",             // dataset path
      "batch_size": 64,                // batch size
      "shuffle": true,                 // shuffle training data before splitting
      "num_workers": 2,                // number of cpu processes to be used for data loading
    }
  },
  "optimizer_G": {                     // optimizer for generator
    "type": "Adam",
    "args":{
      "lr": 2e-4,                     // learning rate
      "weight_decay": 0,               // (optional) weight decay
      "amsgrad": false,
      "betas": [0.5, 0.999]
    }
  },
  "optimizer_D": {                     // optimizer for discriminator
    "type": "Adam",
    "args":{
      "lr": 2e-4,
      "weight_decay": 0,
      "amsgrad": false,
      "betas": [0.5, 0.999]
    }
  },
  "loss": "adversarial_loss",          // loss
  "lr_scheduler_G": {
    "type": "StepLR",                  // learning rate scheduler for generator
    "args": {
      "step_size": 1,
      "gamma": 0.99
    }
  },
  "lr_scheduler_D": {
    "type": "StepLR",                  // learning rate scheduler for discriminator
    "args": {
      "step_size": 1,
      "gamma": 0.99
    }
  },
  "trainer": {
    "type": "GANTrainer",              // trainer class

    "epochs": 100,                     // number of training epochs
    "save_dir": "saved/",              // checkpoints are saved in save_dir/models/name
    "save_period": 1,                  // save checkpoints every save_period epochs
    "verbosity": 2,                    // 0: quiet, 1: per epoch, 2: full

    "visual_tool": "wandb",            // visualization tool
    "api_key_file": "./init/wandb-api-key-file",
    "project": "gan-ada",
    "entity": "gan-augment-project",
  },
  "eval": {
    "save_dir": "saved/generated",
    "n_sample": 1000,
    "batch_size": 8
  },
  "augment": {
    "type": "Ada",                   // augmentation method
    "args": {
      "xflip": 1,
      "rotate90": 1,
      "xint": 1,
      "scale":1,
      "aniso":1,
      "xfrac":1,
      "lumaflip":1,
      "hue":1,
      "saturation": 1,
      "brightness": 1,
      "contrast": 1,
      "cutout": 1
    }
  }
}
```

### Using config files

Modify the configurations in `.json` config files, then run:

```
python train.py --config config.json
```

### Resuming from checkpoints

You can resume from a previously saved checkpoint by:

```
python train.py --resume path/to/checkpoint
```

### Evaluating

Compute statistics of experimental dataset for FID score calculation by:

```
!python dataset_stats.py -nm <output-npz-file-name> -p <path-to-data-dir> -pr <data-portion-taken-into-calculation>
```

Compute FID scores on generated images of each saved checkpoint of a model by:

```
python eval.py -c <path-to-config-file> -r <path-to-checkpoints-dir> -cs <path-to-calculated-stats-file>
```

## Acknowledgements

This project is based on previous work by [victoresque](https://github.com/victoresque) on [PyTorch Template](https://github.com/victoresque/pytorch-template).

