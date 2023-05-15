import importlib
from datetime import datetime
from utils.util import init_wandb
from typing import Dict, Any


class WandB:
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
                message = (
                    "Warning: visualization (WandB) is configured to use, but currently not installed on this "
                    "machine. Please install WandB with 'pip install wandb', set the option in the 'config.yaml' file."
                )

                logger.warning(message)

        # Init writer based on WandB
        self.writer = init_wandb(
            self.writer,
            api_key_file=cfg_trainer["api_key_file"],
            project=cfg_trainer["project"],
            entity=cfg_trainer["entity"],
            name=name,
            config=visualize_config,
        )

        self.step = 0
        self.mode = ""

        self.timer = datetime.now()

    def set_step(self, step, mode="train"):
        self.mode = mode
        self.step = step

        if step == 0:
            self.timer = datetime.now()

        else:
            duration = datetime.now() - self.timer
            self.log({"steps_per_sec": 1 / (duration.total_seconds() + 1e6)})
            self.timer = datetime.now()

    def __getattr__(self, name):
        """
        If visualization is configured to use:
            return add_data() methods of WandB with additional information (step, tag) added.
        Otherwise:
            return a blank function handle that does nothing
        """
        add_data = getattr(self.writer, name, None)

        def wrapper(data: Dict[str, Any], step="Use self.step", *args, **kwargs):
            if add_data is not None:
                # add mode(train/valid) tag
                tag = "{}/{}".format(list(data.keys())[0], self.mode)
                add_data(
                    {tag: data[list(data.keys())[0]]},
                    self.step if step is not None else step,
                    *args,
                    **kwargs
                )

        return wrapper
