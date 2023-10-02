from mmcv.runner import IterBasedRunner, RUNNERS
import torch
from typing import (Any, Callable, Dict, List, Optional, Tuple, Union,
                    no_type_check)
import logging
from mmseg.datasets.pipelines import Compose


@RUNNERS.register_module()
class BilateralFilterRunner(IterBasedRunner):
    def __init__(self,
                 model: torch.nn.Module,
                 batch_processor: Optional[Callable] = None,
                 optimizer: Union[Dict, torch.optim.Optimizer, None] = None,
                 work_dir: Optional[str] = None,
                 logger: Optional[logging.Logger] = None,
                 meta: Optional[Dict] = None,
                 max_iters: Optional[int] = None,
                 max_epochs: Optional[int] = None,
                 pipeline=None) -> None:
        super(BilateralFilterRunner, self).__init__(model, batch_processor,
                                                    optimizer, work_dir, logger,
                                                    meta, max_iters, max_epochs)
        self.pipeline = Compose(pipeline)


    def train(self, data_loader, **kwargs):
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        self._epoch = data_loader.epoch
        data_batch = self.pipeline(next(data_loader))
        data_batch = self.pipeline(data_batch)
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