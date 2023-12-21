import torch
import torch.nn.functional as F

from ..builder import LOSSES
from .base import BaseWeightedLoss


@LOSSES.register_module()
class KLDLoss(BaseWeightedLoss):

    def __init__(self, loss_weight=1.0):
        super().__init__(loss_weight=loss_weight)

    def _forward(self, mu, logvar, labels=None, cluster_centers=None, **kwargs):
        bs = mu.shape[0]
        mu = mu.view(bs, -1)
        logvar = logvar.view(bs, -1)
        if labels is None:
            loss = torch.mean(-0.5 * torch.sum(1 + logvar -
                              mu.pow(2) - logvar.exp(), dim=1), dim=0)
            return loss
        else:
            mu_tgt = torch.index_select(cluster_centers, dim=0, index=labels)
            loss = torch.mean(-0.5 * torch.sum(1 + logvar -
                              (mu - mu_tgt).pow(2) - logvar.exp(), dim=1), dim=0)
            return loss
