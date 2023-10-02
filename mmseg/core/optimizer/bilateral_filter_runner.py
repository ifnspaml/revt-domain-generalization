from mmcv.runner import IterBasedRunner, RUNNERS
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.nn import Parameter
import math
from typing import (Any, Callable, Dict, List, Optional, Tuple, Union,
                    no_type_check)
import logging


def gkern2d(l=21, sig=3):
    """Returns a 2D Gaussian kernel array."""
    ax = np.arange(-l // 2 + 1., l // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2. * sig ** 2))
    return kernel / (1.0 / (2 * math.pi * (sig ** 2)))

# This class is partially adapted from https://github.com/anlcnydn/bilateral/blob/master/bilateral_filter.py
# And modified to work with multiple channels
class Shift(nn.Module):
    def __init__(self, kernel_size=3):
        super(Shift, self).__init__()
        self.kernel_size = kernel_size
        self.pad = self.kernel_size//2

    def forward(self, x):
        n, c, h, w = x.size()
        x_pad = F.pad(x, (self.pad, self.pad, self.pad, self.pad))
        x_pad = x_pad.reshape(n,1,c,h+self.pad*2,w+self.pad*2)

        cat_layers = []
        for y in range(0,self.kernel_size):
            y2 = y+h
            for x in range(0, self.kernel_size):
                x2 = x+w
                xx = x_pad[:,:,:,y:y2,x:x2]
                cat_layers += [xx]
        return torch.cat(cat_layers, 1)


# This class is partially adapted from https://github.com/anlcnydn/bilateral/blob/master/bilateral_filter.py
# And modified to work with multiple channels
class BilateralFilterModule(nn.Module):
    r"""BilateralFilter computes:
        If = 1/W * Sum_{xi C Omega}(I * f(||I(xi)-I(x)||) * g(||xi-x||))
    """

    def __init__(self, channels=3, height=480, width=640, k=7, sigma_space=5, sigma_color=0.1):
        super(BilateralFilterModule, self).__init__()

        self.g = Parameter(torch.Tensor(channels,k*k))
        self.gw = gkern2d(k,sigma_space)
        self.k = k
        gw = np.tile(self.gw.reshape(1,k*k,1,1),(1,1,height,width))
        self.g.data = torch.from_numpy(gw).float()
        #shift
        self.shift = Shift(k)
        self.norm_term_color = (1.0 / (2 * math.pi * (sigma_color ** 2)))
        self.sigma_color = 2*sigma_color**2

    def forward(self, I):
        n, c, h, w = I.size()
        #Is contains the original image mutiple shifted
        Is = self.shift(I).data
        Iex = I.expand(*Is.size())
        D = ((Is-Iex)**2).sum(2)
        g_i = torch.exp(-D / self.sigma_color) * self.norm_term_color
        g_i_s = g_i * self.g.data
        W_denom = torch.sum(g_i_s,dim=1)
        g_i_s = g_i_s.reshape(n, self.k*self.k, 1, h, w).expand((n, self.k*self.k, c, h, w))
        If = torch.sum(g_i_s*Is,dim=1) / W_denom
        return If

class BilateralFilterTorch(object):
    def __init__(self, prob=0.5, max_blur=15, img_size=None, channels=3):
        self.prob = prob
        self.max_blur = max_blur
        self.filter = []
        h,w = img_size
        for k in range(max_blur//2):
            f = BilateralFilterModule(channels,h,w, k=k*2+1, sigma_space=75.0, sigma_color=75.0/255.0)
            f.cuda()
            self.filter.append(f)

    def bilateral_filter(self, img):
        with torch.no_grad():
            c,h,w = img.shape
            img = img.reshape(1, c, h, w)
            img = img.to('cuda')
            blur = np.random.choice(self.max_blur//2)
            color = self.filter[blur](img)
            img_out = color[0]
            return img_out

    def __call__(self, results):
        results["bilateral_filter"] = False
        if np.random.random() < self.prob:
            results["img"] = self.bilateral_filter(results["img"])
            results["bilateral_filter"] = True
            return results
        return results

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
                 prob=0.5, max_blur=15,
                 img_size=None, channels=3) -> None:
        super(BilateralFilterRunner, self).__init__(model, batch_processor,
                                                    optimizer, work_dir, logger,
                                                    meta, max_iters, max_epochs)
        self.filter = BilateralFilterTorch(prob, max_blur, img_size, channels)

    def filter_batch(self, batch):
        for i, img_metas in enumerate(batch["img_metas"].data[0]):
            res_dict = dict()
            res_dict["img"] = batch["img"].data[0][i]
            res_dict = self.filter(res_dict)
            batch["img_metas"].data[0][i]["bilateral_filter"] = res_dict["bilateral_filter"]
            batch["img"].data[0][i] = res_dict["img"]
        return batch

    def train(self, data_loader, **kwargs):
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        self._epoch = data_loader.epoch
        data_batch = next(data_loader)
        data_batch = self.filter_batch(data_batch)
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