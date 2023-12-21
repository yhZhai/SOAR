# Copyright (c) OpenMMLab. All rights reserved.
from .audio_tsn_head import AudioTSNHead
from .base import BaseHead
from .bbox_head import BBoxHeadAVA
from .fbo_head import FBOHead
from .i3d_head import I3DHead
from .lfb_infer_head import LFBInferHead
from .misc_head import ACRNHead
from .roi_head import AVARoIHead
from .slowfast_head import SlowFastHead
from .ssn_head import SSNHead
from .stgcn_head import STGCNHead
from .timesformer_head import TimeSformerHead
from .tpn_head import TPNHead
from .trn_head import TRNHead
from .tsm_head import TSMHead
from .tsn_head import TSNHead
from .x3d_head import X3DHead
from .ae_head import AEHead
from .vae_head import VAEHead
from .cvae_head import CVAEHead
from .cvae_ae_head import CVAEAEHead
from .debias_head import DebiasHead
from .heavy_debias_head import HeavyDebiasHead
from .ae_debias_head import AEDebiasHead
from .i3d_bnn_head import I3DBNNHead
from .i3d_rpl_head import I3DRPLHead
from .slowfast_bnn_head import SlowFastBNNHead
from .slowfast_rpl_head import SlowFastRPLHead
from .tpn_bnn_head import TPNBNNHead
from .tpn_rpl_head import TPNRPLHead
from .tsm_rpl_head import TSMRPLHead
from .tsm_bnn_head import TSMBNNHead

__all__ = [
    'TSNHead', 'I3DHead', 'BaseHead', 'TSMHead', 'SlowFastHead', 'SSNHead',
    'TPNHead', 'AudioTSNHead', 'X3DHead', 'BBoxHeadAVA', 'AVARoIHead',
    'FBOHead', 'LFBInferHead', 'TRNHead', 'TimeSformerHead', 'ACRNHead',
    'STGCNHead', 'AEHead', 'VAEHead',  'CVAEHead', 'CVAEAEHead', 'DebiasHead',
    'HeavyDebiasHead', 'AEDebiasHead', 'I3DBNNHead', 'I3DRPLHead', 'SlowFastBNNHead',
    'SlowFastRPLHead', 'TPNBNNHead', 'TPNRPLHead', 'TSMRPLHead', 'TSMBNNHead'
]
