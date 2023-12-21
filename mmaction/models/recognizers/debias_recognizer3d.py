import torch
from torch import nn
import torch.nn.functional as F

from ..builder import RECOGNIZERS, build_loss
from .recognizer3d import Recognizer3D


@RECOGNIZERS.register_module()
class DebiasRecognizer3D(Recognizer3D):
    """3D recognizer model framework."""

    def __init__(
        self,
        backbone,
        cls_head=None,
        neck=None,
        train_cfg=None,
        test_cfg=None,
        loss_recon=dict(type="MSELoss", loss_weight=0.)
    ):
        super().__init__(backbone, cls_head, neck, train_cfg, test_cfg)

        self.loss_recon = build_loss(loss_recon)

    def forward_train(self, imgs, labels, **kwargs):
        """Defines the computation performed at every call when training."""

        assert self.with_cls_head
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        losses = dict()

        x = self.extract_feat(imgs)
        if self.with_neck:
            x, loss_aux = self.neck(x, labels.squeeze())
            losses.update(loss_aux)

        loss_recon = self.loss(x, imgs, **kwargs)
        losses.update(loss_recon)

        result = self.cls_head(x, imgs)
        gt_labels = labels.squeeze()
        loss_cls = self.cls_head.loss(result, gt_labels, **kwargs)
        losses.update(loss_cls)

        return losses

    def loss(self, outputs, gt, **kwargs):
        losses = dict()
        loss_recon = self.loss_recon(outputs, gt)
        losses["loss_recon"] = loss_recon
        return losses

    def _do_test(self, imgs, return_dict=False):
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
            while view_ptr < total_views:
                batch_imgs = imgs[view_ptr:view_ptr + self.max_testing_views]
                x = self.extract_feat(batch_imgs)
                if self.with_neck:
                    x, _ = self.neck(x)
                feats.append(x)
                view_ptr += self.max_testing_views
            # should consider the case that feat is a tuple
            if isinstance(feats[0], tuple):
                len_tuple = len(feats[0])
                feat = [
                    torch.cat([x[i] for x in feats]) for i in range(len_tuple)
                ]
                feat = tuple(feat)
            else:
                feat = torch.cat(feats)
        else:
            feat = self.extract_feat(imgs)
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
        result = self.cls_head(feat, imgs)
        if isinstance(result, dict):
            cls_score = result["cls_score"]
            cls_score = self.average_clip(cls_score, num_segs)
            result["cls_score"] = cls_score
            if return_dict:
                return result
        else:
            cls_score = self.average_clip(result, num_segs)

        return cls_score

    def forward_test(self, imgs, return_dict=False, **kwargs):
        """Defines the computation performed at every call when evaluation and
        testing."""
        results = self._do_test(imgs, return_dict)
        if not return_dict:
            return results.cpu().numpy()
        return results

    def forward_dummy(self, imgs, softmax=False):
        """Used for computing network FLOPs.

        See ``tools/analysis/get_flops.py``.

        Args:
            imgs (torch.Tensor): Input images.

        Returns:
            Tensor: Class score.
        """
        assert self.with_cls_head
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        x = self.extract_feat(imgs)

        if self.with_neck:
            x, _ = self.neck(x)

        outs = self.cls_head(x, imgs)
        if softmax:
            outs = nn.functional.softmax(outs)
        return (outs, )

    def forward_gradcam(self, imgs):
        """Defines the computation performed at every call when using gradcam
        utils."""
        assert self.with_cls_head
        return self._do_test(imgs)

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
