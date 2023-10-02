from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.core import add_prefix
from mmseg.ops import resize
from .. import builder
from ..builder import SEGMENTORS
from .encoder_decoder import EncoderDecoder
from torch.autograd import Function


# class is copied from: https://github.com/tadeephuy/GradientReversal/blob/master/gradient_reversal/functional.py
class GradientReversal(Function):
    @staticmethod
    def jvp(ctx: Any, *grad_inputs: Any) -> Any:
        raise NotImplementedError("the function jvp is curently not implemented for class GradientReversal")

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x, alpha)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        _, alpha = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = - alpha * grad_output
        return grad_input, None


revgrad = GradientReversal.apply


@SEGMENTORS.register_module()
class EncoderDecoderAugDiscriminator(EncoderDecoder):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self,
                 backbone,
                 decode_head,
                 augmentation_discriminator_head,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 alpha=1.0):
        super(EncoderDecoderAugDiscriminator, self).__init__(backbone, decode_head, neck=neck,
                                                             auxiliary_head=auxiliary_head,
                                                             train_cfg=train_cfg, test_cfg=test_cfg,
                                                             pretrained=pretrained)
        self._init_augmentation_discriminator_head(augmentation_discriminator_head)
        self.alpha = alpha

    def _init_augmentation_discriminator_head(self, augmentation_discriminator_head):
        """Initialize ``decode_head``"""
        self.augmentation_discriminator_head = builder.build_head(augmentation_discriminator_head)

    def _augmentation_discriminator_head_forward_train(self, x, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.augmentation_discriminator_head.forward_train(x, img_metas,
                                                                         gt_semantic_seg,
                                                                         self.train_cfg)

        losses.update(add_prefix(loss_decode, 'discriminator'))
        return losses

    def forward_train(self, img, img_metas, gt_semantic_seg):
        x = self.extract_feat(img)

        losses = dict()

        loss_decode = self._decode_head_forward_train(x, img_metas,
                                                      gt_semantic_seg)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                x, img_metas, gt_semantic_seg)
            losses.update(loss_aux)

        x_rev = []
        alpha = torch.tensor([self.alpha]).cuda()
        for x_i in x:
            x_rev.append(revgrad(x_i, alpha))
        loss_augmentation_discriminator = self._augmentation_discriminator_head_forward_train(x_rev, img_metas,
                                                                                              gt_semantic_seg)
        losses.update(loss_augmentation_discriminator)

        return losses

    def train_step(self, data_batch, optimizer, **kwargs):
        losses = self(**data_batch)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(data_batch['img'].data))

        return outputs
