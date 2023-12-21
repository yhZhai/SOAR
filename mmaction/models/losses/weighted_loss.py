import torch
import torch.nn.functional as F

from ..builder import LOSSES
from .base import BaseWeightedLoss


@LOSSES.register_module()
class WeightedLoss(BaseWeightedLoss):
    """
    This loss modifies L1 loss by applying weight on the reconstruction loss.
    """

    def __init__(
        self,
        loss_weight: float = 1.0,
        loss_impl: str = "mse",
        evidence_type: str = "exp",
        num_class: int = 101,
        sign: float = 1.0,
        detach: bool = False,
        per_batch_norm: bool = False,
        recenter: bool = False,
        weight_src: str = "unc",
    ):
        super().__init__(loss_weight=loss_weight)

        assert loss_impl in ["mse", "l1"], f"unsupported loss type {loss_impl}"
        assert sign in [-1.0, 1.0], f"sign {sign} not in range [-1, 1]"
        assert weight_src in ["unc", "scene"], f"unsupported weight source {weight_src}"

        self.loss_impl = loss_impl
        if evidence_type == "relu":
            self.evidence_fn = F.relu
        elif evidence_type == "exp":
            self.evidence_fn = lambda x: torch.exp(torch.clamp(x, -10, 10))
        elif evidence_type == "softplu":
            self.evidence_fn = F.softplus
        else:
            raise NotImplementedError
        self.num_class = num_class
        self.sign = sign
        self.detach = detach
        self.per_batch_norm = per_batch_norm
        self.recenter = recenter
        self.weight_src = weight_src

    def _forward(self, recon, gt_recon, cls_cam=None, scene_cam=None, **kwargs):
        weight_map = None
        if self.weight_src == "unc":
            evidence = self.evidence_fn(cls_cam)
            alpha = evidence + 1
            uncertainty = self.num_class / torch.sum(alpha, dim=1)
            uncertainty = uncertainty.unsqueeze(1)
            uncertainty = F.interpolate(uncertainty, recon.shape[2:], mode="trilinear")
            uncertainty = self._minmax_normalize(
                uncertainty, per_batch_norm=self.per_batch_norm, recenter=self.recenter
            )
            if self.detach:
                uncertainty = uncertainty.detach()
            weight_map = uncertainty  # b, 1, t, h, w
        elif self.weight_src == "scene":
            scene_cam = F.interpolate(scene_cam, recon.shape[2:], mode="trilinear")
            scene_cam = self._minmax_normalize(
                scene_cam, per_batch_norm=self.per_batch_norm, recenter=self.recenter
            )
            if self.detach:
                scene_cam = scene_cam.detach()
            weight_map = 1 - scene_cam  # b, 1, t, h, w
        else:
            raise NotImplementedError(f"unsupported weight source {self.weight_src}")

        if self.loss_impl == "l1":
            loss = F.l1_loss(recon, gt_recon, reduction="none", **kwargs)
        elif self.loss_impl == "mse":
            loss = F.mse_loss(recon, gt_recon, reduction="none", **kwargs)
        else:
            raise NotImplementedError(f"Unsupported loss type {self.loss_impl}")
        loss = loss * weight_map
        loss = loss.mean() * self.sign
        return loss

    def _minmax_normalize(self, x, per_batch_norm: bool, recenter: bool):
        if per_batch_norm:
            b, c, t, h, w = x.shape
            min_v, _ = x.view(b, -1).min(dim=1, keepdim=True)
            max_v, _ = x.view(b, -1).max(dim=1, keepdim=True)
            mean_v = x.view(b, -1).mean(dim=1, keepdim=True)
            min_v = min_v.view(b, 1, 1, 1, 1).expand_as(x)
            max_v = max_v.view(b, 1, 1, 1, 1).expand_as(x)
            mean_v = mean_v.view(b, 1, 1, 1, 1).expand_as(x)
            x = (x - min_v) / (max_v - min_v)
            if recenter:
                x = x + 1 - mean_v
        else:
            x = (x - x.min()) / (x.max() - x.min())
            if recenter:
                x = x + 1 - x.mean()
        return x
