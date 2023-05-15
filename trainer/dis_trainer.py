import logging

# os.environ["WANDB_SILENT"] = "True"
# os.environ['WANDB_MODE'] = 'offline'
logger = logging.getLogger("wandb")
logger.setLevel(logging.ERROR)
logger.setLevel(logging.WARNING)

import numpy as np
import torch
from torchvision.utils import make_grid
from trainer.base import BaseTrainer
from trainer.metrics.metric import mae, sm
from utils import inf_loop, MetricTracker
from tqdm import tqdm
import wandb


class DISTrainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
        self,
        model,
        criterion,
        metric_ftns,
        optimizer,
        config,
        device,
        data_loader,
        valid_data_loader=None,
        lr_scheduler=None,
        len_epoch=None,
    ):
        super().__init__(model, criterion, metric_ftns, optimizer, config, data_loader)
        self.config = config
        self.device = device
        self.data_loader = data_loader

        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch

        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        # DataFrame metrics
        self.train_metrics = MetricTracker(
            "loss", *[m.__name__ for m in self.metric_ftns], track=self.track
        )
        self.valid_metrics = MetricTracker(
            "loss", *[m.__name__ for m in self.metric_ftns], track=self.track
        )

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """

        tqdm_batch = tqdm(
            iterable=self.data_loader,
            desc="Epoch {}".format(epoch),
            total=len(self.data_loader),
            unit="it",
        )
        self.model.train()
        self.train_metrics.reset()

        for batch_idx, loader in enumerate(tqdm_batch):
            # Load to Device
            if self.device == "cuda:0":
                data = loader["img"].to(device=self.device)
                data = data.type(torch.cuda.FloatTensor)
                mask = loader["mask"].to(device=self.device)
                mask = mask.type(torch.cuda.FloatTensor)

            else:
                data = loader["img"].to(device=self.device)
                data = data.type(torch.FloatTensor)
                mask = loader["mask"].to(device=self.device)
                mask = mask.type(torch.FloatTensor)

            self.optimizer.zero_grad()

            # x_map for metrics, list_maps for loss
            x_map, list_maps = self.model(data)

            loss = self.criterion(list_maps, mask)
            loss.backward()
            self.optimizer.step()

            # Variable for logging
            log_loss = loss.item()

            # Metrics, detach tensor auto-grad to numpy
            if self.device == "cuda:0":
                map_np, mask_np = (
                    x_map.cpu().detach().numpy(),
                    mask.cpu().detach().numpy(),
                )
            else:
                map_np, mask_np = x_map.detach().numpy(), mask.detach().numpy()

            # Metrics
            # log_maxfm, log_wfm = maxfm(map_np, mask_np), wfm(map_np, mask_np)

            log_mae = mae(map_np, mask_np)
            log_sm = sm(map_np, mask_np)

            # Progress bar
            tqdm_batch.set_postfix(loss=log_loss, mae=log_mae, sm=log_sm)

            # WandB
            wandb.log({"loss": log_loss, "mae": log_mae, "sm": log_sm})

            # Logging
            self.track.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update("loss", log_loss)

            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(map_np, mask_np))

            if batch_idx % self.log_step == 0:
                self.logger.debug(
                    "Train Epoch: {} {} Loss: {:.6f}".format(
                        epoch, self._progress(batch_idx), loss.item()
                    )
                )

            del loss, log_loss, log_mae, log_sm, x_map, list_maps

            if batch_idx == self.len_epoch:
                break

        tqdm_batch.close()

        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{"val_" + k: v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()

        with torch.no_grad():
            for batch_idx, loader in enumerate(self.valid_data_loader):
                # Load to Device
                if self.device == "cuda:0":
                    data = loader["img"].to(device=self.device)
                    data = data.type(torch.cuda.FloatTensor)
                    mask = loader["mask"].to(device=self.device)
                    mask = mask.type(torch.cuda.FloatTensor)

                else:
                    data = loader["img"].to(device=self.device)
                    data = data.type(torch.FloatTensor)
                    mask = loader["mask"].to(device=self.device)
                    mask = mask.type(torch.FloatTensor)

                # Forward
                x_fuse, list_maps = self.model(data)
                loss = self.criterion(list_maps, mask)

                # Metrics, detach tensor auto-grad to numpy
                if self.device == "cuda:0":
                    map_np, mask_np = (
                        x_fuse.cpu().detach().numpy(),
                        mask.cpu().detach().numpy(),
                    )
                else:
                    map_np, mask_np = x_fuse.detach().numpy(), mask.detach().numpy()

                # Logging
                self.track.set_step(
                    (epoch - 1) * len(self.valid_data_loader) + batch_idx, "valid"
                )
                self.valid_metrics.update("loss", loss.item())

                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(map_np, mask_np))

                # Log WandB, Predicted will show first
                images = wandb.Image(make_grid(x_fuse[:8], nrow=4))
                self.track.log({"Predicted": images}, step=None)

                # Delete garbage
                del images, loss, x_fuse, list_maps

        # WandB Log Original + GT
        loader = next(iter(self.data_loader))

        self.track.set_step(epoch, "valid")

        # Grid 2 x 4
        original = wandb.Image(make_grid(loader["img"][:8], nrow=4))
        gt = wandb.Image(make_grid(loader["mask"][:8], nrow=4))

        self.track.log({"Original": original}, step=None)
        self.track.log({"Ground Truth": gt}, step=None)

        # Delete garbage
        del original, gt

        # Add histogram of model parameters to the WandB
        for name, p in self.model.named_parameters():
            self.track.add_histogram(name, p, bins="auto")

        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.data_loader, "n_samples"):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
