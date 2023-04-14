import importlib
from datetime import datetime
from utils.util import init_wandb
from typing import Dict, Any

class WandB():
    """
    Weights & Biases makes MLOps easy to track your experiments, manage & version your data, 
    and collaborate with your team so you can focus on building the best models.
    """
    
    def __init__(self, name, cfg_trainer, logger, visual_tool, visualize_config=None):
        self.writer = None
        self.selected_module = ""
        self.name = "wandb"
        
        if visual_tool != "None":
            # Retrieve visualization writer.
            succeeded = False
            
            # Import self.writer = wandb
            try: 
                self.writer = importlib.import_module(visual_tool)
                succeeded = True
                
            except ImportError:
                succeeded = False
                
            self.selected_module = visual_tool

            # Install 
            if not succeeded:
                message = "Warning: visualization (WandB) is configured to use, but currently not installed on this " \
                          "machine. Please install WandB with 'pip install wandb', set the option in the 'config.json' file."
                          
                logger.warning(message)

        # Init writer based on WandB
        self.writer = init_wandb(self.writer, 
                                 api_key_file=cfg_trainer['api_key_file'],
                                 project=cfg_trainer['project'], 
                                 entity=cfg_trainer['entity'],
                                 name=name,
                                 config=visualize_config)

        self.step = 0
        self.mode = ''

        self.timer = datetime.now()

    def set_step(self, step, mode='train'):
        self.mode = mode
        self.step = step
        
        if step == 0:
            self.timer = datetime.now()
            
        else:
            duration = datetime.now() - self.timer
            self.log({'steps_per_sec': 1 / (duration.total_seconds() + 1e6)})
            self.timer = datetime.now()

    def __getattr__(self, name):
        """
        If visualization is configured to use:
            return add_data() methods of TensorBoard with additional information (step, tag) added.
        Otherwise:
            return a blank function handle that does nothing
        """
        add_data = getattr(self.writer, name, None)

        def wrapper(data: Dict[str, Any], step="Use self.step", *args, **kwargs):
            if add_data is not None:
                
                # add mode(train/valid) tag
                tag = '{}/{}'.format(list(data.keys())[0], self.mode)
                add_data({tag: data[list(data.keys())[0]]}, self.step if step is not None else step, *args, **kwargs)

        return wrapper

class TensorboardWriter():
    def __init__(self, log_dir, logger, enabled):
        self.writer = None
        self.selected_module = ""

        if enabled:
            log_dir = str(log_dir)

            # Retrieve vizualization writer.
            succeeded = False
            for module in ["torch.utils.tensorboard", "tensorboardX"]:
                try:
                    self.writer = importlib.import_module(module).SummaryWriter(log_dir)
                    succeeded = True
                    break
                except ImportError:
                    succeeded = False
                self.selected_module = module

            if not succeeded:
                message = "Warning: visualization (Tensorboard) is configured to use, but currently not installed on " \
                    "this machine. Please install TensorboardX with 'pip install tensorboardx', upgrade PyTorch to " \
                    "version >= 1.1 to use 'torch.utils.tensorboard' or turn off the option in the 'config.json' file."
                logger.warning(message)

        self.step = 0
        self.mode = ''

        self.tb_writer_ftns = {
            'add_scalar', 'add_scalars', 'add_image', 'add_images', 'add_audio',
            'add_text', 'add_histogram', 'add_pr_curve', 'add_embedding'
        }
        self.tag_mode_exceptions = {'add_histogram', 'add_embedding'}
        self.timer = datetime.now()

    def set_step(self, step, mode='train'):
        self.mode = mode
        self.step = step
        if step == 0:
            self.timer = datetime.now()
        else:
            duration = datetime.now() - self.timer
            self.add_scalar('steps_per_sec', 1 / duration.total_seconds())
            self.timer = datetime.now()

    def __getattr__(self, name):
        """
        If visualization is configured to use:
            return add_data() methods of tensorboard with additional information (step, tag) added.
        Otherwise:
            return a blank function handle that does nothing
        """
        if name in self.tb_writer_ftns:
            add_data = getattr(self.writer, name, None)

            def wrapper(tag, data, *args, **kwargs):
                if add_data is not None:
                    # add mode(train/valid) tag
                    if name not in self.tag_mode_exceptions:
                        tag = '{}/{}'.format(tag, self.mode)
                    add_data(tag, data, self.step, *args, **kwargs)
            return wrapper
        else:
            # default action for returning methods defined in this class, set_step() for instance.
            try:
                attr = object.__getattr__(name)
            except AttributeError:
                raise AttributeError("type object '{}' has no attribute '{}'".format(self.selected_module, name))
            return attr
