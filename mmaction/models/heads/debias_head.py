# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.cnn import normal_init

from ...core import top_k_accuracy
from ..custom_modules import RevGrad
from ..builder import HEADS, build_loss
from .base import BaseHead


@HEADS.register_module()
class DebiasHead(BaseHead):

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 loss_debias=dict(type='MSELoss'),
                 spatial_type='avg',
                 dropout_ratio=0.5,
                 init_std=0.01,
                 num_scene: int = 2048,
                 # with_bn: bool = False,
                 **kwargs):
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)

        self.spatial_type = spatial_type
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None

        self.filter_module = nn.Sequential(
            # nn.BatchNorm3d(self.in_channels) if with_bn else nn.Identity(),
            nn.LeakyReLU(),
            nn.Conv3d(self.in_channels,
                      self.in_channels,
                      kernel_size=(3, 3, 3),
                      padding=(1, 1, 1)),
            # nn.BatchNorm3d(self.in_channels) if with_bn else nn.Identity(),
            nn.LeakyReLU(),
            nn.Conv3d(self.in_channels,
                      self.in_channels,
                      kernel_size=(3, 3, 3),
                      padding=(1, 1, 1)),
        )
        self.fc_cls = nn.Linear(self.in_channels, self.num_classes)
        self.fc_scene = nn.Sequential(
            RevGrad(),
            nn.Linear(self.in_channels, num_scene)
        )

        if self.spatial_type == 'avg':
            # use `nn.AdaptiveAvgPool3d` to adaptively match the in_channels.
            self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            self.avg_pool = None

        self.loss_debias = build_loss(loss_debias)

    def init_weights(self):
        """Initiate the parameters from scratch."""
        normal_init(self.fc_cls, std=self.init_std)

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        # [N, in_channels, 4, 7, 7]
        x = self.filter_module(x)

        # [N, in_channels, 4, 7, 7]
        if self.avg_pool is not None:
            x = self.avg_pool(x)
        # [N, in_channels, 1, 1, 1]
        if self.dropout is not None:
            x = self.dropout(x)
        # [N, in_channels, 1, 1, 1]
        x = x.view(x.shape[0], -1)
        # [N, in_channels]
        cls_score = self.fc_cls(x)
        scene_score = self.fc_scene(x)
        # [N, num_classes]
        return {
            'cls_score': cls_score,
            'scene_score': scene_score,
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

        loss_scene = self.loss_debias(scene_score, kwargs["scene_feature"])
        losses["loss_scene"] = loss_scene

        return losses
