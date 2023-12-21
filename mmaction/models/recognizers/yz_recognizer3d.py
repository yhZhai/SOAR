from typing import Optional, List
import warnings

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
# from spatial_correlation_sampler import spatial_correlation_sample

from ...core import OutputHook
from ..builder import RECOGNIZERS
from .recognizer3d import Recognizer3D


@RECOGNIZERS.register_module()
class YZRecognizer3D(Recognizer3D):

    """
    format of recon_tgt: {layer}_{property}
        layer set: {frame, layer0, layer1, layer2, layer3}
        property set: {raw, cost, diff}
    """
    def __init__(
        self,
        backbone,
        cls_head=None,
        neck=None,
        train_cfg=None,
        test_cfg=None,
        recon_tgt: Optional[str] = None,
        cost_patch_size: int = 21,
        cost_temperature: float = 0.1,
        cost_impl: str = "cor",
        do_median_filter: bool = False,
        median_win_size: int = 15
    ):

        # reconstruction
        recon_layer, recon_prop = recon_tgt.split("_")
        assert recon_layer in ["frame", "layer0", "layer1", "layer2", "layer3"]
        assert recon_prop in ["raw", "cost", "diff"]
        self.recon_layer = None
        if recon_layer == 'layer3':
            self.recon_layer = 'backbone.layer3.5.conv3.conv'
        elif recon_layer == 'layer2':
            self.recon_layer = 'backbone.layer2.3.conv3.conv'
        elif recon_layer == 'layer1':
            self.recon_layer = 'backbone.layer1.2.conv3.conv'
        elif recon_layer == 'layer0':
            self.recon_layer = 'backbone.conv1.conv'
        elif recon_layer == "frame":
            self.recon_layer = "frame"
        else:
            raise NotImplementedError(f"Reconstruction target {recon_tgt} not "
                                       "supported")
        self.recon_prop = recon_prop
        print(f"Using reconstruction target {self.recon_layer} with {recon_prop}")
        cls_head.update({
            "recon_layer": recon_layer,
            "recon_prop": recon_prop
        })

        # cost volume
        assert cost_impl in ["cor", "dot"], "Only support correlation or dot product for cost volume implementation"
        assert cost_patch_size % 2 == 1
        self.cost_impl = cost_impl
        self.cost_patch_size = cost_patch_size
        self.cost_temperature = cost_temperature
        self.cost_displacement_map = None

        # median filter on raw
        self.do_median_filter = do_median_filter
        if self.do_median_filter:
            assert self.recon_prop == "raw", "Only raw constructions support median filter"
        self.median_win_size = median_win_size

        super().__init__(backbone, cls_head, neck, train_cfg, test_cfg)

    def forward(self, imgs, label=None, return_loss=True,
                return_latent_vector=False, **kwargs):
        """Define the computation performed at every call."""
        if return_latent_vector:  # return ae/vae latent vector
            assert not return_loss
            return self.forward_test(imgs,
                                     return_latent_vector=return_latent_vector,
                                     **kwargs)

        if kwargs.get('gradcam', False):
            del kwargs['gradcam']
            return self.forward_gradcam(imgs, **kwargs)
        if return_loss:
            if label is None:
                raise ValueError('Label should not be None.')
            if self.blending is not None:
                imgs, label = self.blending(imgs, label)
            return self.forward_train(imgs, label, **kwargs)

        return self.forward_test(imgs, **kwargs)

    def _postprocess_recon_tgt(self, recon_tgt):
        # image shape: b, c, t, w, h
        b, c, t, w, h = recon_tgt.shape
        if self.recon_layer == 'frame':  # downsample to 64x64 for computational efficiency
            recon_tgt = F.interpolate(recon_tgt, (t, 64, 64))

        if self.recon_prop == "raw":
            if self.do_median_filter:
                num_frame = recon_tgt.shape[2]
                recon_tgt_list = []
                side_range = int((self.median_win_size - 1) / 2)
                for i in range(num_frame):
                    window = recon_tgt[:, :, max(0, i - side_range): min(num_frame, i + side_range), ...]
                    window = torch.median(window, dim=2, keepdim=True)[0]
                    recon_tgt_list.append(window)
                recon_tgt = torch.cat(recon_tgt_list, dim=2)
            return recon_tgt
        elif self.recon_prop == "diff":
            recon_tgt_t0 = recon_tgt[:, :, :-1, ...]
            recon_tgt_t1 = recon_tgt[:, :, 1:, ...]
            return recon_tgt_t1 - recon_tgt_t0
        elif self.recon_prop == "cost":
            raise NotImplementedError()
            # TODO normalize?
            raw_size = list(recon_tgt.shape[-3:])
            raw_size[0] = raw_size[0] - 1
            recon_tgt = F.max_pool3d(recon_tgt, (1, 2, 2))  # Follow TraDeS
            if self.cost_impl == "cor":
                outputs = []
                for t_cur in range(t - 1):
                    input1 = recon_tgt[:, :, t_cur, ...].contiguous()
                    input2 = recon_tgt[:, :, t_cur + 1, ...].contiguous()
                    # output size: b, cost_patch_size, cost_patch_size, w, h
                    correlation = spatial_correlation_sample(
                        input1,
                        input2,
                        kernel_size=3,
                        patch_size=self.cost_patch_size,
                        stride=1,
                        padding=1,
                        dilation=1,
                        dilation_patch=2
                    )
                    correlation = correlation / self.cost_temperature
                    displacement = torch.arange(-(self.cost_patch_size - 1) // 2,
                                                (self.cost_patch_size - 1) // 2 + 1,
                                                dtype=correlation.dtype,
                                                device=correlation.device)
                    # horizontal
                    horizontal = correlation.sum(dim=1)
                    horizontal = F.softmax(horizontal, dim=1)
                    horizontal = horizontal * displacement[None, :, None, None]
                    horizontal = horizontal.sum(dim=1)
                    # vertical
                    vertical = correlation.sum(dim=2)
                    vertical = F.softmax(vertical, dim=1)
                    vertical = vertical * displacement[None, :, None, None]
                    vertical = vertical.sum(dim=1)

                    result = torch.stack([horizontal, vertical], dim=1)
                    outputs.append(result)

                recon_tgt = torch.stack(outputs, dim=2)

            elif self.cost_impl == "dot":
                outputs = []
                for t_cur in range(t - 1):
                    input1 = recon_tgt[:, :, t_cur, ...].contiguous()
                    input2 = recon_tgt[:, :, t_cur + 1, ...].contiguous()
                    input1 = rearrange(input1, "b c h w -> b c (h w)")
                    input2 = rearrange(input2, "b c h w -> b (h w) c")
                    sim = torch.matmul(input2, input1)  # b, c, (h w), (h w)
                    sim = rearrange(sim, "b s (h w) -> b s h w", h=raw_size[1] // 2)

                    sim_h = sim.max(dim=3)[0]
                    sim_w = sim.max(dim=2)[0]
                    sim_h_softmax = F.softmax(sim_h / self.cost_temperature, dim=2)
                    sim_w_softmax = F.softmax(sim_w / self.cost_temperature, dim=2)
                    if self.cost_displacement_map is None:  # set self.cost_displacement_map if it is none
                        self._get_cost_displacement_map(raw_size[1] // 2, raw_size[2] // 2, recon_tgt.device)
                    off_h = torch.sum(sim_h_softmax * self.cost_displacement_map["v"], dim=2, keepdim=True).permute(0, 2, 1)
                    off_w = torch.sum(sim_w_softmax * self.cost_displacement_map["m"], dim=2, keepdim=True).permute(0, 2, 1)
                    off_h = off_h.view(b, 1, raw_size[1] // 2, raw_size[2] // 2)
                    off_w = off_w.view(b, 1, raw_size[1] // 2, raw_size[2] // 2)

                    result = torch.stack([off_w, off_h], dim=1)
                    outputs.append(result)

                recon_tgt = torch.cat(outputs, dim=2)

            recon_tgt = F.interpolate(recon_tgt, raw_size)
            return recon_tgt
        else:
            raise NotImplementedError

    def _get_cost_displacement_map(self, h: int, w: int, device):
        off_template_w = np.zeros((h, w, w), dtype=np.float32)
        off_template_h = np.zeros((h, w, h), dtype=np.float32)
        for ii in range(h):
            for jj in range(w):
                for i in range(h):
                    off_template_h[ii, jj, i] = i - ii
                for j in range(w):
                    off_template_w[ii, jj, j] = j - jj
        m = np.reshape(off_template_w, newshape=(h * w, w))[None, :, :] * 2
        v = np.reshape(off_template_h, newshape=(h * w, h))[None, :, :] * 2
        self.cost_displacement_map = {
            "m": torch.tensor(m, device=device),
            "v": torch.tensor(v, device=device)
        }

    def forward_train(self, imgs, labels, **kwargs):
        """Defines the computation performed at every call when training."""

        assert self.with_cls_head
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        losses = dict()

        if self.recon_layer != "frame":
            with OutputHook(self, outputs=[self.recon_layer], as_tensor=True) as h:
                x = self.extract_feat(imgs)
                recon_tgt = h.layer_outputs[self.recon_layer]
                recon_tgt = self._postprocess_recon_tgt(recon_tgt)
        else:  # reconstruct frame
            x = self.extract_feat(imgs)
            recon_tgt = self._postprocess_recon_tgt(imgs)

        if self.with_neck:
            x, loss_aux = self.neck(x, labels.squeeze())
            losses.update(loss_aux)

        gt_labels = labels.squeeze()

        gt_recon = recon_tgt.detach()
        loss_dict = None
        outputs = self.cls_head(x, gt_recon)
        loss_dict = self.cls_head.loss(outputs, gt_labels, gt_recon, **kwargs)
        losses.update(loss_dict)

        return losses

    def forward_test(self, imgs, return_latent_vector=False, return_dict=False,
                     **kwargs):
        """Defines the computation performed at every call when evaluation and
        testing."""
        if "get_cams" in kwargs.keys():
            get_cams = kwargs["get_cams"]
        else:
            get_cams = False
        results = self._do_test(imgs, return_latent_vector, return_dict,
                                get_cams=get_cams)
        if not return_dict:
            return results.cpu().numpy()
        return results

    def _do_test(self, imgs, return_latent_vector=False, return_dict=False,
                 get_cams=False):
        """Defines the computation performed at every call when evaluation,
        testing and gradcam."""
        batches = imgs.shape[0]
        num_segs = imgs.shape[1]
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])

        if self.max_testing_views is not None:
            total_views = imgs.shape[0]
            assert num_segs == total_views, (
                'max_testing_views is only compatible '
                'with batch_size == 1')
            view_ptr = 0
            feats = []
            gt_recon = []
            while view_ptr < total_views:
                batch_imgs = imgs[view_ptr:view_ptr + self.max_testing_views]
                if self.recon_layer != "frame":
                    with OutputHook(self, outputs=[self.recon_layer], as_tensor=True) as h:
                        x = self.extract_feat(batch_imgs)
                        recon_tgt = h.layer_outputs[self.recon_layer]
                        recon_tgt = self._postprocess_recon_tgt(recon_tgt)
                else:
                    x = self.extract_feat(imgs)
                    recon_tgt = self._postprocess_recon_tgt(imgs)

                if self.with_neck:
                    x, _ = self.neck(x)
                feats.append(x)
                gt_recon.append(recon_tgt)
                view_ptr += self.max_testing_views
            # should consider the case that feat is a tuple
            if isinstance(feats[0], tuple):
                warnings.warn('Not supported for reconstruction.')
                len_tuple = len(feats[0])
                feat = [
                    torch.cat([x[i] for x in feats]) for i in range(len_tuple)
                ]
                feat = tuple(feat)
            else:
                feat = torch.cat(feats)
                gt_recon = torch.cat(gt_recon)
        else:
            if self.recon_layer != "frame":
                with OutputHook(self, outputs=[self.recon_layer], as_tensor=True) as h:
                    feat = self.extract_feat(imgs)
                    gt_recon = h.layer_outputs[self.recon_layer]
                    gt_recon = self._postprocess_recon_tgt(gt_recon)
            else:
                feat = self.extract_feat(imgs)
                gt_recon = self._postprocess_recon_tgt(imgs)

            if self.with_neck:
                feat, _ = self.neck(feat)

        if self.feature_extraction:
            feat_dim = len(feat[0].size()) if isinstance(feat, tuple) else len(
                feat.size())
            assert feat_dim in [
                5, 2
            ], ('Got feature of unknown architecture, '
                'only 3D-CNN-like ([N, in_channels, T, H, W]), and '
                'transformer-like ([N, in_channels]) features are supported.')
            if feat_dim == 5:  # 3D-CNN architecture
                # perform spatio-temporal pooling
                avg_pool = nn.AdaptiveAvgPool3d(1)
                if isinstance(feat, tuple):
                    feat = [avg_pool(x) for x in feat]
                    # concat them
                    feat = torch.cat(feat, axis=1)
                else:
                    feat = avg_pool(feat)
                # squeeze dimensions
                feat = feat.reshape((batches, num_segs, -1))
                # temporal average pooling
                feat = feat.mean(axis=1)
            return feat

        # should have cls_head if not extracting features
        assert self.with_cls_head
        outputs = self.cls_head(feat, gt_recon=gt_recon, get_cams=get_cams)
        if return_latent_vector:
            latent_vector = outputs['latent_vector']
            latent_vector = self.average_clip(latent_vector, num_segs)
            return latent_vector
        if return_dict:
            outputs['cls_score'] = self.average_clip(outputs['cls_score'], num_segs)
            return outputs

        cls_score = outputs['cls_score']
        cls_score = self.average_clip(cls_score, num_segs)
        return cls_score

    def average_clip(self, cls_score, num_segs=1):
        """Averaging class score over multiple clips.

        Using different averaging types ('score' or 'prob' or None,
        which defined in test_cfg) to computed the final averaged
        class score. Only called in test mode.

        Args:
            cls_score (torch.Tensor): Class score to be averaged.
            num_segs (int): Number of clips for each input sample.

        Returns:
            torch.Tensor: Averaged class score.
        """
        if 'average_clips' not in self.test_cfg.keys():
            raise KeyError('"average_clips" must defined in test_cfg\'s keys')

        average_clips = self.test_cfg['average_clips']
        if average_clips not in ['score', 'prob', 'evidence', None]:
            raise ValueError(f'{average_clips} is not supported. '
                             f'Currently supported ones are '
                             f'["score", "prob", None]')

        if average_clips is None:
            return cls_score

        batch_size = cls_score.shape[0]
        cls_score = cls_score.view(batch_size // num_segs, num_segs, -1)

        if average_clips == 'prob':
            cls_score = F.softmax(cls_score, dim=2).mean(dim=1)
        elif average_clips == 'score':
            cls_score = cls_score.mean(dim=1)
        elif average_clips == 'evidence':
            assert 'evidence_type' in self.test_cfg.keys()
            cls_score = self.evidence_to_prob(cls_score, self.test_cfg['evidence_type'])
            cls_score = cls_score.mean(dim=1)

        return cls_score

    def evidence_to_prob(self, output, evidence_type):
        if evidence_type == 'relu':
            from ..losses.edl_loss import relu_evidence as evidence
        elif evidence_type == 'exp':
            from ..losses.edl_loss import exp_evidence as evidence
        elif evidence_type == 'softplus':
            from ..losses.edl_loss import softplus_evidence as evidence
        alpha = evidence(output) + 1
        S = torch.sum(alpha, dim=-1, keepdim=True)
        prob = alpha / S
        return prob
