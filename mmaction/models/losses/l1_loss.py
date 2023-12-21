import torch.nn.functional as F

from ..builder import LOSSES
from .base import BaseWeightedLoss


@LOSSES.register_module()
class L1Loss(BaseWeightedLoss):

    def __init__(self, loss_weight=1.0):
        super().__init__(loss_weight=loss_weight)

    def _forward(self, recon, gt_recon, **kwargs):
        loss = F.l1_loss(recon, gt_recon, **kwargs)
        return loss
