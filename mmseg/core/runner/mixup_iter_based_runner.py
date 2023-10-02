from mmcv.runner import IterBasedRunner, RUNNERS
import torch
from typing import (Any, Callable, Dict, List, Optional, Tuple, Union,
                    no_type_check)
import logging


@RUNNERS.register_module()
class MixUpIterBasedRunner(IterBasedRunner):
    def __init__(self,
                 model: torch.nn.Module,
                 batch_processor: Optional[Callable] = None,
                 optimizer: Union[Dict, torch.optim.Optimizer, None] = None,
                 work_dir: Optional[str] = None,
                 logger: Optional[logging.Logger] = None,
                 meta: Optional[Dict] = None,
                 max_iters: Optional[int] = None,
                 max_epochs: Optional[int] = None) -> None:
        super(MixUpIterBasedRunner, self).__init__(model, batch_processor,
                                                    optimizer, work_dir, logger,
                                                    meta, max_iters, max_epochs)

    def mixup(self, data_batch, labels):
        alpha = float(torch.rand(1).numpy()[0])
        mixedup_images = (alpha * data_batch + (1 - alpha) * torch.roll(data_batch, 1, 0))
        mixedup_labels = (alpha * labels + (1 - alpha) * torch.roll(labels, 1, 0))
        return mixedup_images, mixedup_labels

    def train(self, data_loader, **kwargs):
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        self._epoch = data_loader.epoch
        data_batch = next(data_loader)

        #mixup images
        for i in range(len(data_batch["img"].data)):
            data_batch["img"].data[i], data_batch["gt_semantic_seg"].data[i] = self.mixup(data_batch["img"].data[i], data_batch["gt_semantic_seg"].data[i])

        self.data_batch = data_batch
        self.call_hook('before_train_iter')
        outputs = self.model.train_step(data_batch, self.optimizer, **kwargs)
        if not isinstance(outputs, dict):
            raise TypeError('model.train_step() must return a dict')
        if 'log_vars' in outputs:
            self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])
        self.outputs = outputs
        self.call_hook('after_train_iter')
        del self.data_batch
        self._inner_iter += 1
        self._iter += 1