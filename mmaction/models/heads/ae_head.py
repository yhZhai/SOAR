from typing import Dict, List, Tuple, Optional
from collections import OrderedDict

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, normal_init, kaiming_init
from einops import rearrange

from ...core import top_k_accuracy
from ..builder import HEADS, build_loss
from ..custom_modules import RevGrad
from .base import BaseHead


@HEADS.register_module()
class AEHead(BaseHead):
    def __init__(
        self,
        num_classes,
        in_channels,
        loss_cls=dict(type="CrossEntropyLoss"),
        loss_recon=dict(type="MSELoss"),
        loss_uncnorm=dict(type="UncNormLoss"),
        spatial_type="avg",
        dropout_ratio=0.5,
        init_std=0.01,
        freeze_cls: bool = False,
        freeze_decoder: bool = False,
        with_bn: bool = True,
        recon_layer: str = "layer3",  # automatically set by Recognizer
        recon_prop: str = "raw",  # automatically set by Recognizer
        recon_grad_rev: bool = False,
        grad_rev_alpha: float = 1.0,
        heavy_cls_head: bool = False,
        do_uncnorm: bool = False,
        **kwargs,
    ):
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)

        self.loss_recon = build_loss(loss_recon)
        self.spatial_type = spatial_type
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None

        if self.spatial_type == "avg":
            # use `nn.AdaptiveAvgPool3d` to adaptively match the in_channels.
            self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            self.avg_pool = None
        if heavy_cls_head:
            self.pre_fc_cls = nn.Sequential(
                nn.LeakyReLU(),
                nn.Conv3d(
                    self.in_channels,
                    self.in_channels,
                    (3, 3, 3),
                    stride=(1, 1, 1),
                    padding=(1, 1, 1),
                ),
                nn.LeakyReLU(),
                nn.Conv3d(
                    self.in_channels,
                    self.in_channels,
                    (3, 3, 3),
                    stride=(1, 1, 1),
                    padding=(1, 1, 1),
                ),
            )
        else:
            self.pre_fc_cls = nn.Identity()
        self.fc_cls = nn.Linear(self.in_channels, self.num_classes)

        assert recon_layer in ["frame", "layer0", "layer1", "layer2", "layer3"]
        assert recon_prop in ["raw", "cost", "diff"]
        self.recon_layer = recon_layer
        self.recon_prop = recon_prop
        self.recon_grad_rev = recon_grad_rev
        if recon_grad_rev:
            self.pre_decoder = RevGrad(alpha=grad_rev_alpha)
        else:
            self.pre_decoder = nn.Identity()
        self.decoder = self._get_decoder(in_channels, with_bn)
        self.freeze_cls = freeze_cls
        self.freeze_decoder = freeze_decoder

        self.do_uncnorm = do_uncnorm
        if self.do_uncnorm:
            self.loss_uncnorm = build_loss(loss_uncnorm)

        self.train()

    def _get_conv_transposed_3d_module(
        self,
        in_channel: int,
        out_channel: int,
        with_bn: bool = True,
        convt_ks: Tuple[int] = (1, 2, 2),
        convt_stride: Tuple[int] = (1, 2, 2),
        convt_padding: Tuple[int] = (0, 0, 0),
        conv_ks: Tuple[int] = (1, 3, 3),
        conv_stride: Tuple[int] = (1, 1, 1),
        conv_padding: Tuple[int] = (0, 1, 1),
        activation=nn.LeakyReLU,
        with_pre_activation: bool = True,
        mid_channel_factor: int = 2,
    ) -> nn.Module:
        mid_channel = int(in_channel / mid_channel_factor)
        module = OrderedDict()

        if with_pre_activation:
            if with_bn:
                module["bn_pre"] = nn.BatchNorm3d(in_channel)
            module["act_pre"] = activation()

        module["convt"] = nn.ConvTranspose3d(
            in_channel,
            in_channel,
            convt_ks,
            stride=convt_stride,
            padding=convt_padding,
        )
        if with_bn:
            module["bn"] = nn.BatchNorm3d(in_channel)
        module["act"] = activation()
        module["conv"] = nn.Conv3d(
            in_channel, mid_channel, conv_ks, stride=conv_stride, padding=conv_padding
        )
        if with_bn:
            module["bn2"] = nn.BatchNorm3d(mid_channel)
        module["act2"] = activation()
        module["conv2"] = nn.Conv3d(
            mid_channel, out_channel, conv_ks, stride=conv_stride, padding=conv_padding
        )

        module = nn.Sequential(module)
        return module

    def _get_decoder(self, in_channels: int, with_bn: bool, reduce_temporal_upsample: bool = False):
        decoder = OrderedDict()
        if self.recon_layer != "frame":
            # if self.recon_prop == "cost":
            #     raise NotImplementedError("cost volume not supported for feature")
            # tgt shape: 1024, 4, 14, 14 = 0.8M
            decoder["stage1"] = self._get_conv_transposed_3d_module(
                in_channels,
                2
                if self.recon_layer == "layer3" and self.recon_prop == "cost"
                else in_channels // 2,
                with_bn,
                convt_ks=(1, 2, 2),
                convt_stride=(1, 2, 2),
                conv_ks=(1, 3, 3),
                conv_stride=(1, 1, 1),
                conv_padding=(0, 1, 1),
            )
            if self.recon_layer in ["layer2", "layer1", "layer0"]:
                # tgt shape: 512, 4, 28, 28 = 1.6M
                decoder["stage2"] = self._get_conv_transposed_3d_module(
                    in_channels // 2,
                    2
                    if self.recon_layer == "layer2" and self.recon_prop == "cost"
                    else in_channels // 4,
                    with_bn,
                    convt_ks=(1, 2, 2),
                    convt_stride=(1, 2, 2),
                    conv_ks=(1, 3, 3),
                    conv_stride=(1, 1, 1),
                    conv_padding=(0, 1, 1),
                )
            if self.recon_layer in ["layer1", "layer0"]:
                # target shape: 256, 8, 56, 56 = 6.4M
                decoder["stage3"] = self._get_conv_transposed_3d_module(
                    in_channels // 4,
                    2
                    if self.recon_layer == "layer1" and self.recon_prop == "cost"
                    else in_channels // 8,
                    with_bn,
                    convt_ks=(2, 2, 2),
                    convt_stride=(2, 2, 2),
                    conv_ks=(3, 3, 3),
                    conv_stride=(1, 1, 1),
                    conv_padding=(1, 1, 1),
                )
            if self.recon_layer in ["layer0"]:
                # target shape: 64, 16, 112, 112 = 12.8M
                decoder["stage4"] = self._get_conv_transposed_3d_module(
                    in_channels // 8,
                    2
                    if self.recon_layer == "layer0" and self.recon_prop == "cost"
                    else in_channels // 32,
                    with_bn,
                    convt_ks=(2, 2, 2),
                    convt_stride=(2, 2, 2),
                    conv_ks=(3, 3, 3),
                    conv_stride=(1, 1, 1),
                    conv_padding=(1, 1, 1),
                )
        else:  # reconstruct frame
            decoder["pool"] = nn.AdaptiveAvgPool3d((4, 4, 4))
            # to 8, 8, 8
            decoder["stage1"] = self._get_conv_transposed_3d_module(
                in_channels,
                in_channels // 4,
                with_bn,
                convt_ks=(2, 2, 2),
                convt_stride=(2, 2, 2),
                conv_ks=(1, 3, 3),
                conv_stride=(1, 1, 1),
                conv_padding=(0, 1, 1),
            )
            # to 16, 16, 16
            decoder["stage2"] = self._get_conv_transposed_3d_module(
                in_channels // 4,
                in_channels // 16,
                with_bn,
                convt_ks=(1, 2, 2) if reduce_temporal_upsample else (2, 2, 2),
                convt_stride=(1, 2, 2) if reduce_temporal_upsample else (2, 2, 2),
                conv_ks=(1, 3, 3),
                conv_stride=(1, 1, 1),
                conv_padding=(0, 1, 1),
            )
            # to 32, 32, 32
            decoder["stage3"] = self._get_conv_transposed_3d_module(
                in_channels // 16,
                in_channels // 64,
                with_bn,
                convt_ks=(1, 2, 2) if reduce_temporal_upsample else (2, 2, 2),
                convt_stride=(1, 2, 2) if reduce_temporal_upsample else (2, 2, 2),
                conv_ks=(1, 3, 3),
                conv_stride=(1, 1, 1),
                conv_padding=(0, 1, 1),
            )
            # to 32, 64, 64
            decoder["stage4"] = self._get_conv_transposed_3d_module(
                in_channels // 64,
                3 if self.recon_prop != "cost" else 2,
                with_bn,
                convt_ks=(1, 2, 2),
                convt_stride=(1, 2, 2),
                conv_ks=(1, 3, 3),
                conv_stride=(1, 1, 1),
                conv_padding=(0, 1, 1),
            )

        decoder = nn.Sequential(decoder)
        return decoder

    def init_weights(self):
        """Initiate the parameters from scratch."""
        normal_init(self.fc_cls, std=self.init_std)
        kaiming_init(self.decoder)

    def forward(self, x, gt_recon=None, get_cams: bool = True):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        result = {}

        # reconstruct
        # [N, in_channels, 4, 7, 7]
        latent_vector = x
        recon = self.pre_decoder(latent_vector)
        recon = self.decoder(recon)
        if self.recon_prop in ["diff", "cost"]:
            recon = recon[:, :, :-1, ...]

        # action classification
        x = self.pre_fc_cls(x)

        # CAM
        if get_cams:
            cls_cam = self.fc_cls(rearrange(x, "b c t h w -> b t h w c"))
            cls_cam = rearrange(cls_cam, "b t h w c -> b c t h w")
        else:
            cls_cam = None
        # for compatibility, set scene_cam as None
        result.update({"cls_cam": cls_cam, "scene_cam": None})

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

        result.update(
            {
                "cls_score": cls_score,
                "recon": recon,
                "latent_vector": latent_vector,
                "gt_recon": gt_recon,
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
        cls_score = outputs["cls_score"]
        recon = outputs["recon"]

        losses = super().loss(cls_score, labels)

        if "weighted" in type(self.loss_recon).__name__.lower():
            loss_recon = self.loss_recon(
                recon, gt_recon, outputs["cls_cam"], outputs["scene_cam"]
            )
        else:
            loss_recon = self.loss_recon(recon, gt_recon)
        losses["loss_recon"] = loss_recon

        if self.do_uncnorm:
            loss_uncnorm = self.loss_uncnorm(recon, gt_recon, outputs["cls_cam"])
            losses["loss_uncnorm"] = loss_uncnorm

        return losses

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_cls:
            self.fc_cls.eval()
            self.fc_cls.requires_grad = False
        if self.freeze_decoder:
            self.decoder.eval()
            self.decoder.requires_grad = False
