import cv2
import numpy as np
from ..builder import PIPELINES
from skimage.util import random_noise


@PIPELINES.register_module()
class MedianFilter(object):
    def __init__(self, prob=0.5, k=7, k_min=2):
        self.prob = prob
        self.k = k
        self.k_min = k_min

    def filter(self, img):
        k = np.random.choice(self.k) * 2 + self.k_min + 1
        color = cv2.medianBlur(img.astype('uint8'), k)
        return color

    def __call__(self, results):
        results["median_filter"] = False
        if np.random.random() < self.prob:
            results["img"] = self.filter(results["img"])
            results["median_filter"] = True
            return results
        return results

@PIPELINES.register_module()
class GaussianBlur(object):
    def __init__(self, prob=0.5, k=7, k_min=2):
        self.prob = prob
        self.k = k
        self.k_min = k_min

    def filter(self, img):
        k = np.random.choice(self.k) * 2 + self.k_min + 1
        color = cv2.GaussianBlur(img.astype('uint8'), (k, k), 0)
        return color

    def __call__(self, results):
        results["gaussian_blur"] = False
        if np.random.random() < self.prob:
            results["img"] = self.filter(results["img"])
            results["gaussian_blur"] = True
            return results
        return results

@PIPELINES.register_module()
class SaltPepperNoise(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def filter(self, img):
        noise_img = (random_noise(img.astype(float)/255.0, mode='s&p')*255).astype('uint8')
        return noise_img

    def __call__(self, results):
        results["salt_pepper_noise"] = False
        if np.random.random() < self.prob:
            results["img"] = self.filter(results["img"])
            results["salt_pepper_noise"] = True
            return results
        return results
