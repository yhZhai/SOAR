import torch.nn.functional as F
import kornia

from ..builder import LOSSES
from .base import BaseWeightedLoss


@LOSSES.register_module()
class SSIMLoss(BaseWeightedLoss):

    def __init__(self, loss_weight: float = 1.0, win_size: int = 7):
        super().__init__(loss_weight=loss_weight)
        self.win_size = win_size

    def _forward(self, recon, gt_recon, **kwargs):
        # shape of recon and gt_recon: b, c, t, h, w
        t = recon.shape[2]
        total_loss = 0
        for i in range(t):
            loss = kornia.losses.ssim_loss(recon[:, :, i, ...],
                                           gt_recon[:, :, i, ...],
                                           window_size=self.win_size, **kwargs)
            total_loss = total_loss + loss
        total_loss = total_loss / t
        return total_loss
