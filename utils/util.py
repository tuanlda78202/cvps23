import os
import glob
import json
import torch
import yaml
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
import yaml

def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def load_yaml(fname):
    fname = Path(fname)
    with fname.open("rt") as file:
        config = yaml.safe_load(file)
    return config

def write_yaml(content, fname):
    fname = Path(fname)
    with fname.open("wt") as handle:
        yaml.dump(content, handle, indent=4, sort_keys=False)
        
def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids

class MetricTracker:
    def __init__(self, *keys, track=None):
        self.track = track
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.track is not None:
            self.track.add_scalar(key, value)
        
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)

def mask_image_list():
    data_dir = os.path.join(os.getcwd(), 'data' + os.sep)
    img_dir = os.path.join("img" + os.sep)
    mask_dir = os.path.join("mask" + os.sep)
    
    img_ext, mask_ext = ".jpg", ".png"
    
    img_list = glob.glob(data_dir + img_dir + "*" + img_ext)
    mask_list = []
    
    for img_path in img_list:
        full_name = img_path.split(os.sep)[-1]
        
        name_ext = full_name.split(".")
        name_list = name_ext[0:-1]
        img_idx = name_list[0]
        
        for i in range(1, len(name_list)):
            img_idx = img_idx + "." + name_list[i]
        
        mask_list.append(data_dir + mask_dir + img_idx + mask_ext)
    
    return img_list, mask_list


def init_wandb(wandb_lib, project, entity, api_key_file='./configs/wandb-api-key-file', dir = "./saved", name=None, config=None):
    """
    Return a new W&B run to be used for logging purposes
    """
    assert os.path.exists(api_key_file), "The given W&B API key file does not exist"
    
    # Set environment API & DIR
    api_key_value = open(api_key_file, "r").read().strip()
    os.environ["WANDB_API_KEY"] = api_key_value
    os.environ["WANDB_DIR"] = dir
    
    # name: user_name in WandB
    return wandb_lib.init(project=project,
                          entity=entity,
                          name=name,
                          config=config)