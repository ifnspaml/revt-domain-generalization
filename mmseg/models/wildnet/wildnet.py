from mmseg.models.wildnet.deepv3 import DeepV3Plus
from ..builder import SEGMENTORS
from ..segmentors import EncoderDecoder
import torch.nn as nn
from argparse import Namespace

class DoNothingHead():
    def __init__(self):
        self.I = nn.Identity()

    def forward_test(self, inputs, img_metas, test_cfg):
        return self.forward(inputs)

    def forward(self, inputs):
        return self.I(inputs)

@SEGMENTORS.register_module()
class WildNet(EncoderDecoder):
    def __init__(self, num_classes,
                 pretrained=None,
                 train_cfg=None,
                 test_cfg=None):
        #super().super().__init__()
        super(EncoderDecoder, self).__init__()
        self.backbone = DeepV3Plus(num_classes, trunk='resnet-50', args=Namespace(fs_layer=[1, 1, 1, 0, 0]))

        self.align_corners = True

        self.decode_head = DoNothingHead()
        self.num_classes = num_classes
        self.freeze_backbone_bn_layers = False

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        assert self.with_decode_head

