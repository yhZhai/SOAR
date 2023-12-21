from typing import Dict

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, normal_init

from mmaction.models.losses import kld_loss

from ...core import top_k_accuracy
from ..builder import HEADS, build_loss
from .base import BaseHead
from .cvae_head import CVAEHead


@HEADS.register_module()
class CVAEAEHead(CVAEHead):

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 loss_recon=dict(type='MSELoss'),
                 loss_kld=dict(type='KLDLoss'),
                 loss_dist=dict(tyep='MSELoss'),
                 spatial_type='avg',
                 dropout_ratio=0.5,
                 init_std=0.01,
                 **kwargs):
        super().__init__(num_classes,
                         in_channels,
                         loss_cls=loss_cls,
                         loss_recon=loss_recon,
                         loss_kld=loss_kld,
                         spatial_type=spatial_type,
                         dropout_ratio=dropout_ratio,
                         init_std=init_std,
                         **kwargs)

        self.ae_encoder = nn.Sequential(
            ConvModule(
                in_channels,
                in_channels,
                (3, 3, 3),
                stride=(1, 2, 2),
                conv_cfg=dict(type='Conv3d'),
                act_cfg=dict(type='ReLU'),
            ),
            ConvModule(
                in_channels,
                in_channels,
                (2, 3, 3),
                stride=(1, 1, 1),
                conv_cfg=dict(type='Conv3d'),
                act_cfg=None
            ),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels * 2,
                in_channels,
                (3, 3, 3),
                stride=(1, 2, 2)
            ),
            nn.ReLU(),
            nn.ConvTranspose3d(
                in_channels,
                in_channels,
                (2, 3, 3),
                stride=(1, 2, 2)
            )
        )
        self.loss_dist = build_loss(loss_dist)

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        # --- begin VAE ---
        # [N, in_channels, 4, 7, 7]
        vae_output = self.encoder(x)
        mu = vae_output[:, :self.in_channels, ...]
        logvar = vae_output[:, self.in_channels:, ...]
        vae_latent = self.reparameterization(mu, logvar)
        mu = mu.squeeze(2).squeeze(2).squeeze(2)
        logvar = logvar.squeeze(2).squeeze(2).squeeze(2)
        # vae_latent = vae_latent.squeeze(2).squeeze(2).squeeze(2)

        ## classification
        # [N, in_channels, 1, 1, 1]
        if self.avg_pool is not None:
            vae_latent = self.avg_pool(vae_latent)
        # [N, in_channels, 1, 1, 1]
        if self.dropout is not None:
            vae_latent = self.dropout(vae_latent)
        # [N, in_channels, 1, 1, 1]
        vae_latent = vae_latent.view(vae_latent.shape[0], -1)
        # [N, in_channels]
        vae_cls_score = self.fc_cls(vae_latent)
        # --- end VAE ---

        # --- begin AE ---
        ae_latent = self.ae_encoder(x)
        ae_latent = ae_latent.squeeze(2).squeeze(2).squeeze(2)
        # --- end AE ---

        # --- begin reconstruction ---
        latent = torch.cat([ae_latent, vae_latent], dim=1)
        latent = latent.unsqueeze(2).unsqueeze(2).unsqueeze(2)
        recon = self.decoder(latent)
        # --- end reconstruction ---

        return {
            'cls_score': vae_cls_score,
            'recon': recon,
            'mu': mu,
            'logvar': logvar,
            'vae_latent': vae_latent,
            'ae_latent': ae_latent
        }

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
        ae_latent = outputs['ae_latent']
        scene_feature = kwargs.pop('scene_feature')

        losses = dict()
        if labels.shape == torch.Size([]):
            labels = labels.unsqueeze(0)
        elif labels.dim() == 1 and labels.size()[0] == self.num_classes \
                and cls_score.size()[0] == 1:
            # Fix a bug when training with soft labels and batch size is 1.
            # When using soft labels, `labels` and `cls_socre` share the same
            # shape.
            labels = labels.unsqueeze(0)

        # --- begin VAE cls ---
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
        # --- end VAE cls ---

        # --- begin VAE KLD ---
        loss_kld = self.loss_kld(mu, logvar)
        losses['loss_kld'] = loss_kld
        # --- end VAE KLD ---

        # --- begin AE distillation ---
        loss_dist = self.loss_dist(ae_latent, scene_feature)
        losses['loss_dist'] = loss_dist
        # --- end AE distillation ---

        # --- begin reconstruction ---
        loss_recon = self.loss_recon(recon, gt_recon)
        losses['loss_recon'] = loss_recon
        # --- end reconstruction ---

        return losses
