import cv2
import numpy as np
from ..builder import PIPELINES
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Parameter
import math


@PIPELINES.register_module()
class BilateralFilter(object):
    def __init__(self, prob=0.5, max_blur=15):
        self.prob = prob
        self.max_blur = max_blur

    def bilateral_filter(self, img):
        blur = np.random.choice(self.max_blur)
        color = cv2.bilateralFilter(img.astype('uint8'), blur, 75, 75)
        return color

    def __call__(self, results):
        results["bilateral_filter"] = False
        if np.random.random() < self.prob:
            results["img"] = self.bilateral_filter(results["img"])
            results["bilateral_filter"] = True
            return results
        return results


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

@PIPELINES.register_module()
class BilateralFilterTorch(object):
    def __init__(self, prob=0.5, max_blur=15, img_size=None, channels=3):
        print("BilateralFilterTorch Constructor called")
        self.prob = prob
        self.max_blur = max_blur
        self.filter = []
        h,w = img_size
        print("BilateralFilterTorch Create Modules")
        for k in range(max_blur//2):
            f = BilateralFilterModule(channels,h,w, k=k*2+1, sigma_space=75.0, sigma_color=75.0)
            print("BilateralFilterTorch move tensors to cuda")
            #f.cuda()
            self.filter.append(f)

    def bilateral_filter(self, img):
        print("BilateralFilterTorch called bilateral_filter")
        with torch.no_grad():
            print(torch.cuda.current_device())
            print(type(img))
            h,w,c = img.shape
            img = img.transpose((2, 0, 1)).reshape(1, c, h, w)
            img = torch.from_numpy(img).float()
            print("BilateralFilterTorch move image to cuda")
            img = img.to('cuda')
            blur = np.random.choice(self.max_blur//2)
            #self.filter[blur].cuda()
            color = self.filter[blur](img)

            print("BilateralFilterTorch finished filtering")
            img_out = color.cpu().detach().numpy()[0]
            img_out = np.transpose(img_out, (1, 2, 0))
            return img_out

    def __call__(self, results):
        print("BilateralFilterTorch called __call__")
        results["bilateral_filter"] = False
        if np.random.random() < self.prob:
            results["img"] = self.bilateral_filter(results["img"])
            results["bilateral_filter"] = True
            return results
        return results
