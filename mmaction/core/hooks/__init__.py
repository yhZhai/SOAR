# Copyright (c) OpenMMLab. All rights reserved.
from .output import OutputHook
from .linear_anneal_kl_weight_hook import LinearAnnealKLWeightHook
from .linear_anneal_recon_weight_hook import LinearAnnealReconWeightHook
from .momentum_update_hook import MomentumUpdateHook
from .anneal_edl_weight_hook import AnnealEDLWeightHook
from .cosine_anneal_adv_recon_weight_hook import CosineAnnealAdvReconWeightHook
from .linear_anneal_grad_rev_alpha_hook import LinearAnnealGradRevAlphaHook

__all__ = ['OutputHook', 'LinearAnnealKLWeightHook',
           'LinearAnnealReconWeightHook', 'MomentumUpdateHook',
           'AnnealEDLWeightHook', 'CosineAnnealAdvReconWeightHook',
           'LinearAnnealGradRevAlphaHook']
