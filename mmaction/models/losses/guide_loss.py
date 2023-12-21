import torch
import torch.nn.functional as F

from ..builder import LOSSES
from .base import BaseWeightedLoss


@LOSSES.register_module()
class GuideLoss(BaseWeightedLoss):

    def __init__(
        self,
        loss_weight: float = 1.0,
        loss_impl: str = "mse",
        evidence_type: str = "exp",
        num_class: int = 101,
        detach_unc: bool = False,
        detach_scene: bool = False,
        per_batch_norm: bool = False,
        do_one_minus: bool = True,
        scene_cam_norm: bool = False,  # default to False due to a bug...
    ):
        super().__init__(loss_weight=loss_weight)

        assert loss_impl in ["mse", "l1"], f"unsupported loss type {loss_impl}"

        self.loss_impl = loss_impl
        if evidence_type == "relu":
            self.evidence_fn = F.relu
        elif evidence_type == "exp":
            self.evidence_fn = lambda x: torch.exp(torch.clamp(x, -10, 10))
        elif evidence_type == "softplu":
            self.evidence_fn = F.softplus
        else:
            raise NotImplementedError(f"unsupported evidence type {evidence_type}")

        self.num_class = num_class
        self.detach_unc = detach_unc
        self.detach_scene = detach_scene
        self.per_batch_norm = per_batch_norm
        self.do_one_minus = do_one_minus
        self.scene_cam_norm = scene_cam_norm

    def _forward(self, cls_cam, scene_cam, **kwargs):
        evidence = self.evidence_fn(cls_cam)
        alpha = evidence + 1
        uncertainty = self.num_class / torch.sum(alpha, dim=1)
        uncertainty = uncertainty.unsqueeze(1)
        uncertainty = self._minmax_normalize(
            uncertainty, per_batch_norm=self.per_batch_norm
        )
        if self.detach_unc:
            uncertainty = uncertainty.detach()

        if self.scene_cam_norm:
            scene_cam = self._minmax_normalize(
                scene_cam, per_batch_norm=self.per_batch_norm
            )
        if self.detach_scene:
            scene_cam = scene_cam.detach()
        if self.do_one_minus:
            scene_cam = 1 - scene_cam

        if self.loss_impl == "l1":
            loss = F.l1_loss(uncertainty, scene_cam, **kwargs)
        elif self.loss_impl == "mse":
            loss = F.mse_loss(uncertainty, scene_cam, **kwargs)
        else:
            raise NotImplementedError(f"Unsupported loss type {self.loss_impl}")
        return loss

    def _minmax_normalize(self, x, per_batch_norm: bool):
        results = []
        if per_batch_norm:
            for b in range(x.shape[0]):
                results.append((x[b] - x[b].min()) / (x[b].max() - x[b].min()))
            results = torch.stack(results, dim=0)
        else:
            results = (x - x.min()) / (x.max() - x.min())
        return results
