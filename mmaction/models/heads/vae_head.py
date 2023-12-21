from typing import Dict

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, normal_init

from mmaction.models.losses import kld_loss
from ...core import top_k_accuracy
from ..builder import HEADS, build_loss
from .base import BaseHead
from .ae_head import AEHead


@HEADS.register_module()
class VAEHead(AEHead):

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 loss_recon=dict(type='MSELoss'),
                 loss_kld=dict(type='KLDLoss'),
                 spatial_type='avg',
                 dropout_ratio=0.5,
                 init_std=0.01,
                 freeze_cls: bool = False,
                 recon_tgt: str = None,
                 with_bn: bool = False,
                 **kwargs):
        super().__init__(num_classes, in_channels, loss_cls, loss_recon,
                         spatial_type, dropout_ratio, init_std, freeze_cls,
                         recon_tgt, with_bn, **kwargs)

        self.loss_kld = build_loss(loss_kld)
        self.mu_layer = nn.Conv3d(self.in_channels, self.in_channels, kernel_size=(1, 1, 1))
        self.logvar_layer = nn.Conv3d(self.in_channels, self.in_channels, kernel_size=(1, 1, 1))

    def reparameterization(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        x = mu + eps * std
        return x

    def forward(self, x, gt_recon=None):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        # reconstruct
        # [N, in_channels, 4, 7, 7]
        mu = self.mu_layer(x)
        logvar = self.logvar_layer(x)
        if self.training:
            latent_vector = self.reparameterization(mu, logvar)
        else:
            latent_vector = mu
        recon = self.decoder(latent_vector)

        # classification
        # [N, in_channels, 1, 1, 1]
        if self.avg_pool is not None:
            x = self.avg_pool(x)
        # [N, in_channels, 1, 1, 1]
        if self.dropout is not None:
            x = self.dropout(x)
        # [N, in_channels]
        x = x.view(x.shape[0], -1)
        # [N, num_classes]
        cls_score = self.fc_cls(x)

        result = {
            'cls_score': cls_score,
            'recon': recon,
            'latent_vector': latent_vector,
            'gt_recon': gt_recon,
            'mu': mu,
            'logvar': logvar
        }
        return result

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

        loss_kld = self.loss_kld(mu, logvar)
        losses['loss_kld'] = loss_kld

        # for log purpose
        losses['mu'] = mu.mean()
        losses['logvar'] = logvar.mean()

        return losses
