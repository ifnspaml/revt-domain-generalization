from mmcv.runner.hooks import HOOKS, OptimizerHook
import torch.nn as nn
import torch

from mmcv.utils import (_BatchNorm)


# copied from "https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/optimizer.py"
@HOOKS.register_module()
class MaxWeightMSDOptimizerHook(OptimizerHook):
    """Optimizer Hook implements multi-iters gradient cumulating.
    Args:
        cumulative_iters (int, optional): Num of gradient cumulative iters.
            The optimizer will step every `cumulative_iters` iters.
            Defaults to 1.
    Examples:
        #>>> # Use cumulative_iters to simulate a large batch size
        #>>> # It is helpful when the hardware cannot handle a large batch size.
        #>>> loader = DataLoader(data, batch_size=64)
        #>>> optim_hook = GradientCumulativeOptimizerHook(cumulative_iters=4)
        #>>> # almost equals to
        #>>> loader = DataLoader(data, batch_size=256)
        #>>> optim_hook = OptimizerHook()
    """

    def __init__(self, weight_space_distance_factor: float = 0.5, gamma: float = 0.2, beta: float = 2,
                 **kwargs):
        super().__init__(**kwargs)

        self.weight_space_distance_factor = weight_space_distance_factor
        self.gamma = gamma
        self.beta = beta

    def after_train_iter(self, runner):
        runner.optimizer.zero_grad()

        inverse_weight_distance_loss = 1.0 / torch.pow(runner.outputs['weight_distance'] + self.beta,
                                                       self.gamma) * self.weight_space_distance_factor
        #inverse_weight_distance_loss.retain_grad()
        #print(f"IWDL: {inverse_weight_distance_loss}, WD: {runner.outputs['weight_distance']}")
        loss = runner.outputs['loss'] * (1 - self.weight_space_distance_factor) + inverse_weight_distance_loss

        loss.backward()
        #print(f"inverse_weight_distance_loss gradient: {inverse_weight_distance_loss.grad}")

        if self.grad_clip is not None:
            grad_norm = self.clip_grads(runner.model.parameters())
            if grad_norm is not None:
                # Add grad norm to the logger
                runner.log_buffer.update({'grad_norm': float(grad_norm)},
                                         runner.outputs['num_samples'])
        runner.optimizer.step()
