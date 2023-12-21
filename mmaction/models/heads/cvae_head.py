from typing import Dict

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, normal_init

from mmaction.models.losses import kld_loss

from ...core import top_k_accuracy
from ..builder import HEADS, build_loss
from .base import BaseHead
from .vae_head import VAEHead


@HEADS.register_module()
class CVAEHead(VAEHead):

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 loss_recon=dict(type='MSELoss'),
                 loss_kld=dict(type='KLDLoss'),
                 spatial_type='avg',
                 dropout_ratio=0.5,
                 init_std=0.01,
                 **kwargs):
        super().__init__(num_classes, in_channels, loss_cls, loss_recon,
                         loss_kld, spatial_type, dropout_ratio, init_std,
                         **kwargs)

        # used for training
        self.cluster_centers = nn.parameter.Parameter(torch.randn(num_classes,
                                                                  in_channels))

        # used for saving previous cluster centers
        prev_cluster_centers = torch.rand_like(self.cluster_centers.data)
        prev_cluster_centers.data = self.cluster_centers.data
        self.register_buffer('prev_cluster_centers', prev_cluster_centers)

    def loss(self, outputs: Dict, labels, gt_recon, **kwargs):
        """Calculate the loss given output ``cls_score``, target ``labels``.

        Args:
            cls_score (torch.Tensor): The output of the model.
            labels (torch.Tensor): The target output of the model.

        Returns:
            dict: A dict containing field 'loss_cls'(mandatory)
            and 'topk_acc'(optional).
        """
        cls_score = outputs['cls_score']
        recon = outputs['recon']
        mu = outputs['mu']
        logvar = outputs['logvar']

        losses = dict()
        if labels.shape == torch.Size([]):
            labels = labels.unsqueeze(0)
        elif labels.dim() == 1 and labels.size()[0] == self.num_classes \
                and cls_score.size()[0] == 1:
            # Fix a bug when training with soft labels and batch size is 1.
            # When using soft labels, `labels` and `cls_socre` share the same
            # shape.
            labels = labels.unsqueeze(0)

        if not self.multi_class and cls_score.size() != labels.size():
            top_k_acc = top_k_accuracy(cls_score.detach().cpu().numpy(),
                                       labels.detach().cpu().numpy(),
                                       self.topk)
            for k, a in zip(self.topk, top_k_acc):
                losses[f'top{k}_acc'] = torch.tensor(
                    a, device=cls_score.device)

        elif self.multi_class and self.label_smooth_eps != 0:
            labels = ((1 - self.label_smooth_eps) * labels +
                      self.label_smooth_eps / self.num_classes)

        loss_cls = self.loss_cls(cls_score, labels, **kwargs)
        # loss_cls may be dictionary or single tensor
        if isinstance(loss_cls, dict):
            losses.update(loss_cls)
        else:
            losses['loss_cls'] = loss_cls

        loss_recon = self.loss_recon(recon, gt_recon)
        losses['loss_recon'] = loss_recon

        loss_kld = self.loss_kld(mu, logvar, labels, self.cluster_centers)
        losses['loss_kld'] = loss_kld

        return losses
