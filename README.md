# Salient Object Detection for Korean Name Card
This is the source code for the project "Salient Object Detection for Korean Name Card" of the course "Computer Vision" Summer 2023.

## Abstract 
Currently, existing image segmentation tasks mainly focus on segmenting objects with specific characteristics, e.g., salient, camouflaged, meticulous, or specific categories. Most of them have the same input/output formats, and barely use exclusive mechanisms designed for segmenting targets in their models, which means almost all tasks are dataset-dependent. Thus, it is very promising to formulate a category-agnostic DIS task for accurately segmenting objects with different structure complexities, regardless of their characteristics. Compared with semantic segmentation, the proposed DIS task usually focuses on images with single or a few targets, from which getting richer accurate details of each target is more feasible. 

In this project, we will investigate the powerful of salient object detection in the real world by experimenting it over a various methods to see whether and how it works with Korean Name Card dataset.

- [Salient Object Detection for Korean Name Card](#salient-object-detection-for-korean-name-card)
  - [Project Structure](#project-structure)
  - [Usage](#usage)
    - [Config file format](#config-file-format)
    - [Using config files](#using-config-files)
    - [Resuming from checkpoints](#resuming-from-checkpoints)
    - [Evaluating](#evaluating)
    - [Inference](#inference)
    - [Web Interface](#web-interface)

<!-- /code_chunk_output -->

## Project Structure

```
CVP/
├── scripts/ - bash script to experiments
|
├── tools/ - script to training, testing, inference and web interface
|
├── trainer/ - trainer classes 
|
├── model/ - architectures, losses and metrics
|
├── base/ - abstract base classes
│   
├── configs/ - training config
│
├── data/ - storing input data
|
├── data_loader/ - custom dataset and dataloader
│
├── saved/ - trained models config, log-dir and logging output
│
├── logger/ - module for tensorboard visualization and logging
|
├── utils/ - utility functions
```

## Usage

Install the required packages:

```
pip install -r requirements.txt
```

Running private repository on Kaggle:
1. [Generate your token](https://github.com/settings/tokens)
2. Get repo address from `github.com/.../...git`: 
```bash
git clone https://your_personal_token@your_repo_address.git
cd CVP
```
### Config file format

<details>
<summary>Config files are in JSON format</summary>

```javascript
{
    "name": "U2NetFull_scratch_1gpu-bs8_KNC_size512",
    "n_gpu": 1,
  
    "arch": {
      "type": "u2net_full",
      "args": {}
    },

    "data_loader": {
      "type": "KNC_DataLoader",
      "args": {
        "batch_size": 8,
        "shuffle": true,
        "num_workers": 1,
        "validation_split": 0.1,
        "output_size": 320,
        "crop_size": 288
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


    "metrics": [
      "pixel_accuracy", "dice", "precision", "recall"
    ],


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
  
      "tensorboard": false,
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

</details>

### Using config files
Modify the configurations in `.json` config files, then run:

```bash
bash scripts/u2net_train.sh [CONFIG] [BATCH_SIZE] [EPOCHS]
```

### Resuming from checkpoints
You can resume from a previously saved checkpoint by:

```bash
bash scripts/u2net_train.sh --resume path/to/the/ckpt
```

### Evaluating
```bash
python tools/eval.py
```

### Inference 
Running demo on notebook `inference.ipynb` in [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tuanlda78202/CVP/)


### Web Interface 
```bash
python tools/app.py
```
