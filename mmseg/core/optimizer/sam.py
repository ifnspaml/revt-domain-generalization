from mmcv.runner.optimizer import OPTIMIZER_BUILDERS, OPTIMIZERS
from mmcv.runner.hooks import HOOKS, Hook
from mmcv.runner import IterBasedRunner, RUNNERS
from torch.optim import Optimizer, SGD, AdamW
from mmseg.core.optimizer.sam_src.sam import SAM
from typing import Iterable, Union, Callable, Optional, List

@RUNNERS.register_module()
class ClosureIterBasedRunner(IterBasedRunner):
    def train(self, data_loader, **kwargs):
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        self._epoch = data_loader.epoch
        data_batch = next(data_loader)
        self.call_hook('before_train_iter')
        def forward_pass():
            outputs = self.model.train_step(data_batch, self.optimizer, **kwargs)
            if not isinstance(outputs, dict):
                raise TypeError('model.train_step() must return a dict')
            if 'log_vars' in outputs:
                self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])

            self.outputs = outputs
            return outputs['loss']

        self._forward_pass = forward_pass
        self._forward_pass()

        self.call_hook('after_train_iter')
        self._inner_iter += 1
        self._iter += 1



@HOOKS.register_module()
class OptimizerClosureHook(Hook):

    def __init__(self, grad_clip=None):
        self.grad_clip = grad_clip

    def clip_grads(self, params):
        params = list(
            filter(lambda p: p.requires_grad and p.grad is not None, params))
        if len(params) > 0:
            return clip_grad.clip_grad_norm_(params, **self.grad_clip)

    def after_train_iter(self, runner):
        def closure():
            runner._forward_pass()
            runner.optimizer.zero_grad()
            runner.outputs['loss'].backward()
            if self.grad_clip is not None:
                grad_norm = self.clip_grads(runner.model.parameters())
                if grad_norm is not None:
                    # Add grad norm to the logger
                    runner.log_buffer.update({'grad_norm': float(grad_norm)},
                                             runner.outputs['num_samples'])
            return runner.outputs['loss']
        runner.optimizer.zero_grad()
        runner.outputs['loss'].backward()

        if self.grad_clip is not None:
            grad_norm = self.clip_grads(runner.model.parameters())
            if grad_norm is not None:
                # Add grad norm to the logger
                runner.log_buffer.update({'grad_norm': float(grad_norm)},
                                         runner.outputs['num_samples'])

        runner.optimizer.step(closure)


@OPTIMIZERS.register_module()
class SharpnessAwareMinimization(Optimizer):

    def __init__(self, params=None, rho=0.05, subtype="SGD", **kwargs):
        if subtype == "AdamW":
            base_optimizer = AdamW
        elif subtype == "SGD":
            base_optimizer = SGD
        else:
            raise NotImplementedError(f"The type {subtype} is not implemented for StochasticWeightAveraging")

        self.opt = SAM(params, base_optimizer, rho, **kwargs)
        self.param_groups = self.opt.param_groups

    def __setstate__(self, state):
        return self.opt.__setstate__(state)

    def state_dict(self):
        return self.opt.state_dict()

    def load_state_dict(self, state_dict):
        return self.opt.load_state_dict(state_dict)

    def zero_grad(self, set_to_none: bool=...):
        return self.opt.zero_grad(set_to_none)

    def step(self, closure: Optional[Callable[[], float]]=...):
        #print("closure: ", closure)
        return self.opt.step(closure)

    def add_param_group(self, param_group: dict):
        return self.opt.add_param_group(param_group)

