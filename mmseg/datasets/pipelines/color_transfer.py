import cv2
import numpy as np
from ..builder import PIPELINES
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image

@PIPELINES.register_module()
class ImageNetColorTransfer(object):
    def __init__(self, imagenet_path, prob=0.5):
        self.prob = prob

        self.imagenet_set = datasets.ImageFolder(
            imagenet_path
        )

    def get_mean_and_std(self, x):
        x_mean, x_std = cv2.meanStdDev(x)
        x_mean = np.hstack(np.around(x_mean, 2))
        x_std = np.hstack(np.around(x_std, 2))
        return x_mean, x_std

    def color_transfer(self, t, s):
        convert_tensor = transforms.ToTensor()
        t = np.array(t)
        t = cv2.cvtColor(t, cv2.COLOR_RGB2LAB)
        s = cv2.cvtColor(np.array(s), cv2.COLOR_RGB2LAB)

        t_mean, t_std = self.get_mean_and_std(t)
        s_mean, s_std = self.get_mean_and_std(s)
        t = (t - t_mean) * (s_std / t_std) + s_mean
        t = np.around(t).clip(0, 255).astype("uint8")
        return cv2.cvtColor(t, cv2.COLOR_LAB2RGB)

    def __call__(self, results):
        if np.random.random() < self.prob:
            rnd_idx = np.random.choice(len(self.imagenet_set))
            color_src_image, _ = self.imagenet_set[rnd_idx]
            results["img"] = self.color_transfer(results["img"], color_src_image)
            return results
        return results
