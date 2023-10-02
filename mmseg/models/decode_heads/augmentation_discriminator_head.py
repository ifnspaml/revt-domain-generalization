import torch.nn as nn
import torch
from ..builder import HEADS


@HEADS.register_module()
class AugmentationDiscriminatorHead(nn.Module):
    def __init__(self, keys=("flip", "photometric", "pixmix")):
        super().__init__()
        self.keys = keys
        self.cross_entropy_loss = nn.MSELoss()
        self.c1 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.c2 = nn.Conv2d(128+64, 128, kernel_size=3, stride=2, padding=1)
        self.c3 = nn.Conv2d(320+128, 320, kernel_size=3, stride=2, padding=1)
        self.c4 = nn.Conv2d(512+320, 512, kernel_size=3, stride=2, padding=1)
        self.max_pool = nn.MaxPool2d(2, stride=2)
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(512*16, 1000)
        self.relu1 = nn.ReLU()
        self.dense2 = nn.Linear(1000, len(keys))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        c1, c2, c3, c4 = x
        _c1 = self.c1(c1)
        _c2 = torch.cat([_c1, c2], dim=1)
        _c2 = self.c2(_c2)
        _c3 = torch.cat([_c2, c3], dim=1)
        _c3 = self.c3(_c3)
        _c4 = torch.cat([_c3, c4], dim=1)
        _c4 = self.c4(_c4)
        x = self.max_pool(_c4)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.relu1(x)
        x = self.dense2(x)
        x = self.sigmoid(x)
        return x

    def get_label(self, img_metas):
        img_metas = img_metas if isinstance(img_metas, list) else [img_metas]
        l_imgs = []
        for img_meta in img_metas:
            l_img = []
            for key in self.keys:
                if img_meta[key]:
                    l_img.append(1)
                else:
                    l_img.append(0)
            l_imgs.append(l_img)
        return torch.tensor(l_imgs, dtype=torch.float).cuda()

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        output = self.forward(inputs)
        label = self.get_label(img_metas)
        losses = self.losses(output, label)
        return losses

    def losses(self, pred, label):
        output = self.cross_entropy_loss(pred, label)
        loss = dict()
        loss['discriminator_head_loss'] = output
        return loss
