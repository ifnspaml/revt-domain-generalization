import mmcv
import numpy as np
from mmcv.utils import deprecated_api_warning, is_tuple_of
from numpy import random

from ..builder import PIPELINES
from IPython import embed
from .transforms import PhotoMetricDistortion
import copy
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch

def get_ab(beta):
  if np.random.random() < 0.5:
    a = np.float32(np.random.beta(beta, 1))
    b = np.float32(np.random.beta(1, beta))
  else:
    a = 1 + np.float32(np.random.beta(1, beta))
    b = -np.float32(np.random.beta(1, beta))
  return a, b

def add(img1, img2, beta):
  a,b = get_ab(beta)
  img1, img2 = img1 * 2 - 255, img2 * 2 - 255
  out = np.add(a * img1, b * img2)
  return (out + 255) / 2

def multiply(img1, img2, beta):
  a,b = get_ab(beta)
  img1, img2 = img1 * 2, img2 * 2
  out = (img1 ** a) * (img2.clip(1e-37) ** b)
  return out / 2



@PIPELINES.register_module()
class PixMix(object):
    def __init__(self, mixing_set_path, img_scale, k=4, beta=3):
        self.k = k
        self.beta = beta
        self.mixings = [add, multiply]
        self.augmentation = []
        self.augmentation.append(PhotoMetricDistortion())

        self.mixing_set = datasets.ImageFolder(
            mixing_set_path,
            transform=transforms.Compose([
                transforms.Resize(img_scale[0] + 64),
                transforms.RandomCrop(img_scale[0])
            ])
        )


    def augment_input(self, results):
        aug = np.random.choice(self.augmentation)
        return aug(results)

    def pix_mix_augmentation(self, image_dict, mixing_pic):
        if np.random.random() < 0.5:
            mixed = self.augment_input(copy.deepcopy(image_dict))
        else:
            mixed = copy.deepcopy(image_dict)
            mixed['PhotoMetricDistortionBrightness'] = False
            mixed['PhotoMetricDistortionContrast'] = False
            mixed['PhotoMetricDistortionSaturation'] = False
            mixed['PhotoMetricDistortionHue'] = False

        for _ in range(np.random.randint(self.k + 1)):

            aug_image_copy = self.augment_input(copy.deepcopy(image_dict))
            if np.random.random() < 0.5:
                convert_tensor = transforms.ToTensor()
                aug_image_copy["img"] = convert_tensor(mixing_pic).permute(1,2,0).numpy()

            mixed["pixmix"] = True
            mixed_op = np.random.choice(self.mixings)
            mixed["img"] = mixed_op(mixed["img"], aug_image_copy["img"], self.beta)
            mixed["img"] = np.clip(mixed["img"], 0, 255)

        return mixed

    def __call__(self, results):
        rnd_idx = np.random.choice(len(self.mixing_set))
        mixing_pic, _ = self.mixing_set[rnd_idx]
        results["pixmix"] = False
        return self.pix_mix_augmentation(results, mixing_pic)