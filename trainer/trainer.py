import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from model.metric import pixel_accuracy, dice, precision, recall
from utils import inf_loop, MetricTracker
from tqdm import tqdm

class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config,
                 device, data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        
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
        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        tqdm_batch = tqdm(iterable=self.data_loader, 
                          desc="Epoch {}".format(epoch),
                          total=len(self.data_loader),
                          unit="it")

        self.model.train()
        self.train_metrics.reset()
        
        for batch_idx, loader in enumerate(tqdm_batch):            
            
            # Load to Device 
            data, mask = loader["img"].to(self.device), loader["mask"].to(self.device)

            if self.device == "cuda":
                data = data.type(torch.cuda.FloatTensor)
                mask = mask.type(torch.cuda.FloatTensor)
            else:
                data = data.type(torch.FloatTensor)
                mask = mask.type(torch.FloatTensor)
            
            self.optimizer.zero_grad()
            
            # x_map for metrics, list_maps for loss 
            x_map, list_maps = self.model(data)
                
            loss0, loss = self.criterion(list_maps, mask)            
            loss.backward()
            self.optimizer.step()

            # Progress bar
            tqdm_batch.set_postfix(loss=loss.item(),
                                   pixel_accuracy=pixel_accuracy(mask, x_map),
                                   dice = dice(mask, x_map),
                                   precision = precision(mask, x_map),
                                   recall = recall(mask, x_map))
            
            # Logging
            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(mask, x_map))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(epoch,
                                                                           self._progress(batch_idx),
                                                                           loss.item()))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break
        
        tqdm_batch.close()
                    
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

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
                data, mask = loader["img"].to(self.device), loader["mask"].to(self.device)
                
                x_map, list_maps = self.model(data)
                loss0, loss = self.criterion(list_maps, mask)      

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(mask, x_map))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # Add histogram of model parameters to the Tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)


