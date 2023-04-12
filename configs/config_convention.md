# Config Name Style

We follow the below style to name config files. Contributors are advised to follow the same style.

```python
{algorithm name}_{model component names [component1]_[component2]_[...]}_{training settings}_{training dataset information}_{testing dataset information}
```

The file name is divided to five parts. All parts and components are connected with `_` and words of each part or component should be connected with `-`.

* {`algorithm` `name`}: The name of the algorithm, such as `deeplabv3`, `pspnet`, etc.
  

* {`model` `component` `names`}: Names of the components used in the algorithm such as backbone, head, etc. For example, `r50-d8` means using ResNet50 backbone and use output of backbone is 8 times downsampling as input.
  

* {`training` `settings`}: Information of training settings such as batch size, augmentations, loss, learning rate scheduler, and epochs/iterations. For example: `4xb4-ce-linearlr-40K` means using 4-gpus x 4-images-per-gpu, CrossEntropy loss, Linear learning rate scheduler, and train 40K iterations. Some abbreviations:
  * {`gpu` x `batch_per_gpu`}: GPUs and samples per GPU. `bN` indicates N batch size per GPU. E.g. `8xb2` is the short term of 8-gpus x 2-images-per-gpu. And 4xb4 is used by default if not mentioned.
  * {`schedule`}: training schedule, options are `20k`, `40k`, etc. `20k` and `40k` means 20000 iterations and 40000 iterations respectively.


* {`training` `dataset` `information`}: Training dataset names like `cityscapes`, `ade20k`, etc, and input resolutions. For example: `cityscapes-768x768` means training on `cityscapes` dataset and the input shape is `768x768`.
  

* {`testing` `dataset` `information`} (optional): Testing dataset name for models trained on one dataset but tested on another. If not mentioned, it means the model was trained and tested on the same dataset type.

**Example**: `configs/pspnet/pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py`