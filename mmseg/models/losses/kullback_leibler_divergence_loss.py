import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES

@LOSSES.register_module()
class KullbackLeiblerDivergence(nn.Module):

    def __init__(self,
                 reduction='mean',
                 use_sigmoid=True,
                 loss_weight=1):
        super(KullbackLeiblerDivergence, self).__init__()
        self.reduction = reduction
        self.use_sigmoid = True
        self.loss_weight = loss_weight

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function."""
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        cls_score = F.log_softmax(cls_score, dim=1)
        label = F.log_softmax(label, dim=1)
        loss = F.kl_div(
            cls_score,
            label,
            reduction=reduction,
            log_target=True)
        return loss * self.loss_weight
