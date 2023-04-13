# Salient Object Detection for Korean Namecard

This is the source code for the project "Salient Object Detection for Korean Namecard" of the course "Computer Vision" Summer 2023.

In this project, we will investigate the powerful of salient object detection in the real world by experimenting it over a various methods to see whether and how it works with Korean namecard dataset.

<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 ordere  dList=false} -->

<!-- code_chunk_output -->

- [Salient Object Detection for Korean Namecard](#salient-object-detection-for-korean-namecard)
  - [Project Structure](#project-structure)
  - [Usage](#usage)
    - [Config file format](#config-file-format)
    - [Using config files](#using-config-files)
    - [Resuming from checkpoints](#resuming-from-checkpoints)
    - [Evaluating](#evaluating)
    - [Web Interface](#web-interface)
  - [Acknowledgements](#acknowledgements)

<!-- /code_chunk_output -->

## Project Structure

```
cvp/
│
├── train.py - main script to start training
├── eval.py - script to compute evaluation score on each saved checkpoint of a specified model
│
├── parse_config.py - class to handle config file and cli options
│
├── base/ - abstract base classes
│   ├── base_data_loader.py
│   ├── base_rsu.py
│   └── base_trainer.py
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

Running private repository on Kaggle:
1. [Generate your token](https://github.com/settings/tokens)
2. Get repo address under code -> HTTPS -> your_repo_address=start from `github.com/…../..git` to the end

```
!git clone https://your_personal_token@your_repo_address.git
%cd CVP
!bash scripts/u2net.sh [CONFIG] [BATCH_SIZE] [EPOCHS]
```
### Config file format

Config files are in `.json` format:

```javascript
{
    "name": "U2NetFull_KNC_b64",
    "n_gpu": 1,
  
    "arch": {
      "type": "u2net_full",
      "args": {}
    },

    "data_loader": {
      "type": "KNC_DataLoader",
      "args": {
        "data_dir": "data",
        "batch_size": 8,
        "shuffle": true,
        "num_workers": 1,
        "validation_split": 0.1
      }
    },
  
    "optimizer": {
      "type": "Adam",
      "args": {
        "lr": 1e-3,
        "weight_decay": 0,
        "eps": 1e-08,
        "betas": [0.9, 0.999]
      }
    },

    
    "loss": "multi_bce_fusion",


    "metrics": ["accuracy", "top_k_acc"],


    "lr_scheduler": {
      "type": "StepLR",
      "args": {
        "step_size": 50,
        "gamma": 0.1
      }
    },


    "trainer": {
      "type": "Trainer",
  
      "epochs": 50,

      "save_dir": "saved/",
      "save_period": 5,
      "verbosity": 1,
  
      "visual_tool": "wandb",
      "__comment_1.1": "torch.utils.tensorboard",
      "__comment_1.2": "tensorboardX",
      "__comment_1.3": "wandb",
      "__comment_1.4": "None",
      "api_key_file": "./wandb-api-key-file",
      "project": "knc",
      "entity": "cvp-knc",
      "name": "test",
      "__comment_2.1": "Set name for one running"
    },


    "test": {
      "save_dir": "saved/generated",
      "n_sample": 2000,
      "batch_size": 32
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
```
python test.py
```

### Web Interface 
```python
python app.py
```

## Acknowledgements

This project is based on previous work by [victoresque](https://github.com/victoresque) on [PyTorch Template](https://github.com/victoresque/pytorch-template).

