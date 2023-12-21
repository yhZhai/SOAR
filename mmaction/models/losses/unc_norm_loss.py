import torch
import torch.nn.functional as F

from ..builder import LOSSES
from .base import BaseWeightedLoss


@LOSSES.register_module()
class UncNormLoss(BaseWeightedLoss):
    """
    This loss applies to the uncertainty map.
    """
    def __init__(
        self,
        loss_weight=1.0,
        evidence_type: str = "exp",
        num_class: int = 101,
        k: float = 0.125,
        sign: float = 1,
        self_norm: bool = False
    ):
        super().__init__(loss_weight=loss_weight)

        if evidence_type == "relu":
            self.evidence_fn = F.relu
        elif evidence_type == "exp":
            self.evidence_fn = lambda x: torch.exp(torch.clamp(x, -10, 10))
        elif evidence_type == "softplu":
            self.evidence_fn = F.softplus
        else:
            raise NotImplementedError
        self.num_class = num_class
        assert 0.5 >= k > 0
        self.k = k
        assert sign == 1. or sign == -1.
        self.sign = sign
        self.self_norm = self_norm

    def _forward(self, recon, gt_recon, cls_cam=None, **kwargs):
        evidence = self.evidence_fn(cls_cam)
        alpha = evidence + 1
        uncertainty = self.num_class / torch.sum(alpha, dim=1)
        uncertainty = uncertainty.unsqueeze(1)

        if not self.self_norm:
            # whether use L1 or MSE does not matter, as they are both monotonic functions
            loss = F.l1_loss(recon, gt_recon, reduction="none", **kwargs)
            loss = loss.mean(dim=1, keepdim=True)
            loss = F.interpolate(loss, uncertainty.shape[2:])
            loss = loss.flatten(start_dim=1)

        bs = recon.shape[0]
        uncertainty = uncertainty.flatten(start_dim=1)
        num_sample = int(self.k * uncertainty.shape[1])

        # uncertainty and reconstruction error should be inversely correlated,
        # as high uncertainty indicates low-freqency background part, which has
        # low reconstruction error
        loss_norm = 0
        for b in range(bs):
            if self.self_norm:
                topk_indices = torch.topk(uncertainty[b, ...], k=num_sample)[1]
                bottomk_indices = torch.topk(uncertainty[b, ...], k=num_sample, largest=False)[1]
            else:
                topk_indices = torch.topk(loss[b, ...], k=num_sample)[1]
                bottomk_indices = torch.topk(loss[b, ...], k=num_sample, largest=False)[1]
            # I want this to be low, i.e., certain
            topk_unc = torch.index_select(uncertainty[b, ...], dim=0, index=topk_indices)
            # I want this to be high, i.e., uncertain
            bottomk_unc = torch.index_select(uncertainty[b, ...], dim=0, index=bottomk_indices)
            loss_norm = loss_norm + self.sign * (topk_unc.mean() - bottomk_unc.mean())

        return loss_norm
