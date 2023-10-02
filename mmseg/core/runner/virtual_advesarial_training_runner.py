from mmcv.runner import IterBasedRunner, RUNNERS
import torch
from typing import (Any, Callable, Dict, List, Optional, Tuple, Union,
                    no_type_check)
import logging
from mmcv.parallel import DataContainer as DC
from mmseg.models.builder import build_loss


@RUNNERS.register_module()
class VirtualAdversarialTrainingRunner(IterBasedRunner):
    def __init__(self,
                 model: torch.nn.Module,
                 batch_processor: Optional[Callable] = None,
                 optimizer: Union[Dict, torch.optim.Optimizer, None] = None,
                 work_dir: Optional[str] = None,
                 logger: Optional[logging.Logger] = None,
                 meta: Optional[Dict] = None,
                 max_iters: Optional[int] = None,
                 max_epochs: Optional[int] = None,
                 iter_d=1,
                 radius=1,
                 vat_loss=dict(type='KullbackLeiblerDivergence', loss_weight=1.0)) -> None:
        super(VirtualAdversarialTrainingRunner, self).__init__(model, batch_processor,
                                                    optimizer, work_dir, logger,
                                                    meta, max_iters, max_epochs)
        self.n_power = iter_d
        self.XI = 1e-6
        self.epsilon = radius
        self.vat_loss = build_loss(vat_loss)

    def forward_pass(self, data_batch, **kwargs):
        #for img, gt_semantic_seg in zip(data_batch["img"].data, data_batch["gt_semantic_seg"].data):
        #    pass

        seg_logits = [self.get_logits(b, data_batch["img_metas"]) for b in data_batch["img"].data]
        losses = [self.get_model_loss(logits, label) for logits, label in zip(seg_logits, data_batch["gt_semantic_seg"].data)]

        for i, l in enumerate(losses):
            if self.model.module.with_auxiliary_head:
                loss_aux = self.model.module._auxiliary_head_forward_train(
                    data_batch["img"].data[i], data_batch["img_metas"], data_batch["gt_semantic_seg"].data[i])
                l.update(loss_aux)

            if 'log_vars' in losses:
                self.log_buffer.update(losses['log_vars'], losses['num_samples'])

        self.outputs = losses[0] if len(losses) == 1 else losses
        self.seg_logits = seg_logits

    def get_offseted_batch(self, batch, input_offset):
        new_batch = batch.copy()
        img = []

        for i, _ in enumerate(input_offset):
            img.append(new_batch["img"].data[i].cuda() + input_offset[i][0].cuda())

        new_batch["img"] = DC(img)

        return new_batch

    def get_normalized_vector(self, d):
        return [torch.nn.functional.normalize(d_.view(d_.size(0), -1), p=2, dim=1).reshape(d_.size()) for d_ in d]

    def get_logits(self, img, metas):
        return self.model.module.encode_decode(img.cuda(), metas)

    def get_model_loss(self, logits, gt_semantic_seg):
        losses = self.model.module.decode_head.losses(logits, gt_semantic_seg.cuda())
        loss, log_vars = self.model.module._parse_losses(losses)

        outputs = dict(
            loss=loss,
            log_vars=log_vars)

        return outputs

    def generate_virtual_adversarial_perturbation(self, data_batch, logits, **kwargs):
        d = []
        for i in range(len(data_batch["img"].data)):
            d.append(torch.randn_like(data_batch["img"].data[i]).cuda())

        for _ in range(self.n_power):
            d = self.get_normalized_vector(d)
            d = [self.XI * d_.requires_grad_() for d_ in d]
            offseted_batch = self.get_offseted_batch(batch=data_batch, input_offset=d)

            logits_r = [self.get_logits(b, offseted_batch["img_metas"]) for b in offseted_batch["img"].data]
            dist = [self.vat_loss(l, l_r) for l, l_r in zip(logits, logits_r)]

            grad = [torch.autograd.grad(di, [d_])[0] for di, d_ in zip(dist, d)]
            for i in range(len(d)):
                d[i] = grad[i].detach()
        return [self.epsilon * self.get_normalized_vector(d_) for d_ in d]

    def train(self, data_loader, **kwargs):
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        self._epoch = data_loader.epoch
        data_batch = next(data_loader)
        self.data_batch = data_batch
        self.call_hook('before_train_iter')

        self.forward_pass(data_batch, **kwargs)

        r_vadv = self.generate_virtual_adversarial_perturbation(data_batch, self.seg_logits, **kwargs)

        offseted_batch = self.get_offseted_batch(batch=data_batch, input_offset=r_vadv)
        logits_radv = [self.get_logits(b, offseted_batch["img_metas"]) for b in offseted_batch["img"].data]
        vat_loss = [self.vat_loss(l, l_r) for l, l_r in zip(self.seg_logits, logits_radv)]
        vat_loss = vat_loss[0] if len(vat_loss) == 0 else vat_loss
        model_loss = self.outputs['loss']
        added_loss = model_loss + vat_loss[0]

        self.outputs['loss'] = added_loss

        self.call_hook('after_train_iter')

        del self.data_batch

        self._inner_iter += 1
        self._iter += 1
