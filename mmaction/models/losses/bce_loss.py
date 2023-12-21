import torch.nn.functional as F

from ..builder import LOSSES
from .base import BaseWeightedLoss


@LOSSES.register_module()
class BCELoss(BaseWeightedLoss):

    def __init__(self, loss_weight=1.0, do_norm=False):
        super().__init__(loss_weight=loss_weight)
        self.do_norm = do_norm

    def _forward(self, recon, gt_recon, **kwargs):
        if self.do_norm:
            max_v = max(recon.max(), gt_recon.max()).detach()
            min_v = min(recon.min(), gt_recon.min()).detach()
            recon = (recon - min_v) / (max_v - min_v)
            gt_recon = (gt_recon - min_v) / (max_v - min_v)
        loss = F.binary_cross_entropy(recon, gt_recon, **kwargs)
        return loss
