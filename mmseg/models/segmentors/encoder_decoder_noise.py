import torch
from ..builder import SEGMENTORS
from .encoder_decoder import EncoderDecoder


@SEGMENTORS.register_module()
class EncoderDecoderNoise(EncoderDecoder):
    """Encoder Decoder segmentors.

    EncoderDecoderNoise typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    Different from the base class EncoderDecoder, this class adds gaussian noise to the
    features before they are fed into the head.
    """

    def __init__(self,
                 backbone,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 epsillon=0.03):
        super(EncoderDecoderNoise, self).__init__(backbone, decode_head, neck=neck, auxiliary_head=auxiliary_head,
              train_cfg=train_cfg, test_cfg=test_cfg, pretrained=pretrained)
        self.epsillon = epsillon

    def forward_train(self, img, img_metas, gt_semantic_seg):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        x = self.extract_feat(img)
        for i, x_i in enumerate(x):
            noise = torch.normal(0, self.epsillon, size=x_i.size()).cuda()
            x[i] = x[i] + noise
        losses = dict()

        loss_decode = self._decode_head_forward_train(x, img_metas,
                                                      gt_semantic_seg)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                x, img_metas, gt_semantic_seg)
            losses.update(loss_aux)

        return losses
