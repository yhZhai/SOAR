# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from einops import rearrange
from mmcv.cnn import normal_init

from ...core import top_k_accuracy
from ..custom_modules import RevGrad
from ..builder import HEADS, build_loss, build_backbone
from .base import BaseHead


@HEADS.register_module()
class HeavyDebiasHead(BaseHead):

    def __init__(self,
                 num_classes,
                 in_channels,
                 cls_backbone,
                 scene_backbone,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 loss_debias=dict(type='MSELoss'),  # change to CE loss to do scene classification
                 spatial_type='avg',
                 dropout_ratio=0.5,
                 init_std=0.01,
                 scene_frame_sample_num: int = 4,
                 grad_rev_alpha: float = 1.,
                 num_scene_classes: int = 365,
                 **kwargs):
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)

        assert 32 % scene_frame_sample_num == 0, f"{32 % scene_frame_sample_num} is not zero"

        self.cls_backbone = build_backbone(cls_backbone)
        scene_backbone = build_backbone(scene_backbone)
        self.scene_backbone = nn.Sequential(
            RevGrad(alpha=grad_rev_alpha),
            scene_backbone
        )

        self.spatial_type = spatial_type
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None

        self.fc_cls = nn.Linear(self.in_channels, self.num_classes)

        if self.spatial_type == 'avg':
            # use `nn.AdaptiveAvgPool3d` to adaptively match the in_channels.
            self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            self.avg_pool = None

        self.do_scene_cls = loss_debias["type"] == "CrossEntropyLoss"
        self.loss_debias = build_loss(loss_debias)
        self.scene_frame_sample_num = scene_frame_sample_num
        if self.do_scene_cls:
            print(f"doing scene classifcation with {loss_debias['type']}")
            self.fc_scene_cls = nn.Linear(self.in_channels, num_scene_classes)
        else:
            print(f"doing scene feature distillation with {loss_debias['type']}")
            self.fc_scene_cls = nn.Identity()

    def init_weights(self):
        """Initiate the parameters from scratch."""
        normal_init(self.fc_cls, std=self.init_std)
        normal_init(self.fc_scene_cls, std=self.init_std)

    def forward(self, x, raw_frames=None):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        recon = x

        cls_feature = self.cls_backbone(x)
        # [N, in_channels, 4, 7, 7]
        if self.avg_pool is not None:
            cls_feature = self.avg_pool(cls_feature)
        # [N, in_channels, 1, 1, 1]
        if self.dropout is not None:
            cls_feature = self.dropout(cls_feature)
        # [N, in_channels, 1, 1, 1]
        cls_feature = cls_feature.view(cls_feature.shape[0], -1)
        # [N, in_channels]
        cls_score = self.fc_cls(cls_feature)

        b, c, t = x.shape[:3]
        interval = t // self.scene_frame_sample_num
        sel_frame = torch.arange(0, t, interval)
        x = x[:, :, sel_frame]
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        scene_feature = self.scene_backbone(x)
        scene_feature = rearrange(scene_feature, '(b t) c h w -> b c t h w', b=b)
        if self.avg_pool is not None:
            scene_feature = self.avg_pool(scene_feature)
        scene_feature = scene_feature.view(scene_feature.shape[0], -1)
        # [N, in_channels]
        scene_score = self.fc_scene_cls(scene_feature)

        return {
            'cls_score': cls_score,
            'scene_score': scene_score,
            'recon': recon,
            'gt_recon': raw_frames
        }

    def loss(self, result, labels, **kwargs):
        """Calculate the loss given output ``cls_score``, target ``labels``.

        Args:
            cls_score (torch.Tensor): The output of the model.
            labels (torch.Tensor): The target output of the model.

        Returns:
            dict: A dict containing field 'loss_cls'(mandatory)
            and 'topk_acc'(optional).
        """
        cls_score = result["cls_score"]
        scene_score = result["scene_score"]
        losses = super().loss(cls_score, labels)

        if self.do_scene_cls:
            loss_scene = self.loss_debias(scene_score, kwargs["scene_pred"])
        else:
            loss_scene = self.loss_debias(scene_score, kwargs["scene_feature"])
        losses["loss_scene"] = loss_scene

        return losses
