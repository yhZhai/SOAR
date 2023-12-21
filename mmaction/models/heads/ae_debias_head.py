from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from ..custom_modules import RevGrad
from ..builder import HEADS, build_loss
from .ae_head import AEHead


@HEADS.register_module()
class AEDebiasHead(AEHead):
    def __init__(
        self,
        num_classes,
        in_channels,
        loss_cls=...,
        loss_recon=...,
        loss_uncnorm=...,
        spatial_type="avg",
        dropout_ratio=0.5,
        init_std=0.01,
        freeze_cls: bool = False,
        freeze_decoder: bool = False,
        with_bn: bool = True,
        recon_layer: str = "layer3",
        recon_prop: str = "raw",
        recon_grad_rev: bool = False,
        recon_grad_rev_alpha: float = 1,
        heavy_cls_head: bool = False,
        do_uncnorm: bool = False,
        loss_debias=dict(type="CrossEntropyLoss"),
        num_scene_classes: int = 365,
        scene_grad_rev_alpha: float = 1,
        do_guide: bool = False,
        loss_guide=dict(type="GuideLoss"),
        reduce_temporal_upsample: bool = False,
        **kwargs,
    ):
        super().__init__(
            num_classes,
            in_channels,
            loss_cls,
            loss_recon,
            loss_uncnorm,
            spatial_type,
            dropout_ratio,
            init_std,
            freeze_cls,
            freeze_decoder,
            with_bn,
            recon_layer,
            recon_prop,
            recon_grad_rev,
            recon_grad_rev_alpha,
            heavy_cls_head,
            do_uncnorm,
            **kwargs,
        )
        self.fc_scene_cls = nn.Sequential(
            RevGrad(alpha=scene_grad_rev_alpha),
            nn.Linear(in_channels, in_channels),
            nn.LeakyReLU(),
            nn.Linear(in_channels, in_channels),
            nn.LeakyReLU(),
            nn.Linear(in_channels, in_channels),
            nn.LeakyReLU(),
            nn.Linear(in_channels, in_channels),
            nn.LeakyReLU(),
            nn.Linear(in_channels, num_scene_classes),
        )
        self.loss_debias = build_loss(loss_debias)
        self.do_guide = do_guide
        self.loss_guide = build_loss(loss_guide)
        self.decoder = self._get_decoder(in_channels, with_bn, reduce_temporal_upsample=reduce_temporal_upsample)

    def forward(self, x, gt_recon=None, get_cams: bool = True):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        result = {}
        # slowfast
        if isinstance(x, tuple):
            assert len(x) == 2  # slow and fast pathways
            x = list(x)
            x[0] = F.interpolate(x[0], size=[4, 7, 7])
            x[1] = F.interpolate(x[1], size=[4, 7, 7])
            x = torch.cat(x, dim=1)

        # reconstruct
        # [N, in_channels, 4, 7, 7]
        latent_vector = x
        recon = self.pre_decoder(latent_vector)
        recon = self.decoder(recon)
        if self.recon_prop in ["diff", "cost"]:
            recon = recon[:, :, :-1, ...]

        # action and scene classification
        x = self.pre_fc_cls(x)

        # CAM
        if get_cams:
            cls_cam = self.fc_cls(rearrange(x, "b c t h w -> b t h w c"))
            scene_cam = self.fc_scene_cls(rearrange(x, "b c t h w -> b t h w c"))
            cls_cam = rearrange(cls_cam, "b t h w c -> b c t h w")
            scene_cam = rearrange(scene_cam, "b t h w c -> b c t h w")
        else:
            cls_cam = None
            scene_cam = None
        result.update({"cls_cam": cls_cam, "scene_cam": scene_cam})

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
        scene_score = self.fc_scene_cls(x)

        result.update(
            {
                "cls_score": cls_score,
                "recon": recon,
                "latent_vector": latent_vector,
                "gt_recon": gt_recon,
                "scene_score": scene_score,
                "feature": x,
            }
        )
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
        scene_label = kwargs["scene_pred"].argmax(dim=1)
        scene_cam = []
        for i in range(scene_label.shape[0]):
            scene_cam.append(outputs["scene_cam"][i, scene_label[i], None, ...])
        scene_cam = torch.stack(scene_cam, dim=0)
        outputs["scene_cam"] = scene_cam

        losses = super().loss(outputs, labels, gt_recon, **kwargs)

        scene_score = outputs["scene_score"]
        loss_scene = self.loss_debias(scene_score, kwargs["scene_pred"])
        losses["loss_scene"] = loss_scene

        if self.do_guide:
            loss_guide = self.loss_guide(outputs["cls_cam"], outputs["scene_cam"])
            losses["loss_guide"] = loss_guide

        return losses
