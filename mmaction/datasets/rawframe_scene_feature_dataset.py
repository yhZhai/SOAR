from typing import Dict
import copy
import os.path as osp

import numpy as np
import torch
from scipy.special import softmax
from rich import print

from mmaction.datasets.pipelines import Resize
from .builder import DATASETS
from .rawframe_dataset import RawframeDataset


@DATASETS.register_module()
class RawframeSceneFeatureDataset(RawframeDataset):

    def __init__(self,
                 ann_file,
                 pipeline,
                 data_prefix=None,
                 test_mode=False,
                 filename_tmpl='img_{:05}.jpg',
                 with_offset=False,
                 multi_class=False,
                 num_classes=None,
                 start_index=1,
                 modality='RGB',
                 sample_by_class=False,
                 power=0.,
                 dynamic_length=False,
                 scene_feature_root=None,
                 scene_pred_root=None,
                 **kwargs):
        super().__init__(
            ann_file,
            pipeline,
            data_prefix,
            test_mode,
            filename_tmpl,
            with_offset,
            multi_class,
            num_classes,
            start_index,
            modality,
            sample_by_class=sample_by_class,
            power=power,
            dynamic_length=dynamic_length,
            **kwargs
        )
        assert osp.exists(scene_feature_root), f"scene feature root {scene_feature_root} is not a valid path"
        assert osp.exists(scene_pred_root), f"scene prediction root {scene_pred_root} is not a valid path"
        self.scene_feature_root = scene_feature_root
        self.scene_pred_root = scene_pred_root

    def get_scene_feature_and_pred(self, results: Dict):
        # scene feature
        relative_path = results['frame_dir'].split('/')[-2:]
        relative_path = '/'.join(relative_path)
        scene_feature_path = self.scene_feature_root + '/' + relative_path + '.npy'
        if osp.exists(scene_feature_path):
            scene_feature = np.load(scene_feature_path)
        else:
            print(f"[red]using random scene feature, {scene_feature_path} does not exist[/red]")
            scene_feature = np.random.rand(1, 2048)

        ## convert to torch tensor
        scene_feature = torch.tensor(scene_feature)
        scene_feature = scene_feature.squeeze(0)

        # scene prediction
        scene_pred_path = self.scene_pred_root + '/' + relative_path + '.npy'
        if osp.exists(scene_pred_path):
            scene_pred = np.load(scene_pred_path)
        else:
            print(f"[red]using random scene prediction, {scene_pred_path} does not exist[/red]")
            scene_pred = np.random.rand(1, 365)
        scene_pred = softmax(scene_pred, axis=1)

        ## convert to torch tensor
        scene_pred = torch.tensor(scene_pred)
        scene_pred = scene_pred.squeeze(0)

        return {
            "scene_feature": scene_feature,
            "scene_pred": scene_pred
        }

    def prepare_train_frames(self, idx):
        """Prepare the frames for training given the index."""

        def pipeline_for_a_sample(idx):
            results = copy.deepcopy(self.video_infos[idx])
            results['filename_tmpl'] = self.filename_tmpl
            results['modality'] = self.modality
            results['start_index'] = self.start_index

            # prepare tensor in getitem
            if self.multi_class:
                onehot = torch.zeros(self.num_classes)
                onehot[results['label']] = 1.
                results['label'] = onehot

            raw_out = self.pipeline(results)

            scene_info = self.get_scene_feature_and_pred(results)
            raw_out.update(scene_info)
            return raw_out

        if isinstance(idx, tuple):
            index, short_cycle_idx = idx
            last_resize = None
            for trans in self.pipeline.transforms:
                if isinstance(trans, Resize):
                    last_resize = trans
            origin_scale = self.default_s
            long_cycle_scale = last_resize.scale

            if short_cycle_idx in [0, 1]:
                # 0 and 1 is hard-coded as PySlowFast
                scale_ratio = self.short_cycle_factors[short_cycle_idx]
                target_scale = tuple(
                    [int(round(scale_ratio * s)) for s in origin_scale])
                last_resize.scale = target_scale
            res = pipeline_for_a_sample(index)
            last_resize.scale = long_cycle_scale
            return res
        else:
            return pipeline_for_a_sample(idx)

    def prepare_test_frames(self, idx):
        """Prepare the frames for testing given the index."""
        results = copy.deepcopy(self.video_infos[idx])
        results['filename_tmpl'] = self.filename_tmpl
        results['modality'] = self.modality
        results['start_index'] = self.start_index

        # prepare tensor in getitem
        if self.multi_class:
            onehot = torch.zeros(self.num_classes)
            onehot[results['label']] = 1.
            results['label'] = onehot

        raw_out = self.pipeline(results)

        scene_info = self.get_scene_feature_and_pred(results)
        raw_out.update(scene_info)
        return raw_out
