from mmcv.runner.optimizer import OPTIMIZER_BUILDERS, OPTIMIZERS
from torch.optim import Optimizer, SGD, AdamW
from torchcontrib.optim import SWA
from typing import Iterable, Union, Callable, Optional, List


@OPTIMIZERS.register_module()
class StochasticWeightAveraging(Optimizer):

    def __init__(self, swa_start=10, swa_freq=5, swa_lr=0.05, subtype="AdamW", params=None, **kwargs):
        base_opt = None
        if subtype == "AdamW":
            base_opt = AdamW(params,** kwargs)
        elif subtype == "SGD":
            base_opt = SGD(params, **kwargs)
        else:
            raise NotImplementedError(f"The type {subtype} is not implemented for StochasticWeightAveraging")

        self.param_groups = base_opt.param_groups
        self.opt = SWA(base_opt, swa_start=swa_start, swa_freq=swa_freq, swa_lr=swa_lr)

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
        return self.opt.step()

    def add_param_group(self, param_group: dict):
        return self.opt.add_param_group(param_group)

