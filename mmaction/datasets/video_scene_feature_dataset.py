from typing import Dict
import os.path as osp
import copy

import numpy as np
import torch
from scipy.special import softmax
from rich import print

from mmaction.datasets.pipelines import Resize
from .video_dataset import VideoDataset
from .builder import DATASETS


@DATASETS.register_module()
class VideoSceneFeatureDataset(VideoDataset):
    """Video dataset for action recognition.

    The dataset loads raw videos and apply specified transforms to return a
    dict containing the frame tensors and other information.

    The ann_file is a text file with multiple lines, and each line indicates
    a sample video with the filepath and label, which are split with a
    whitespace. Example of a annotation file:

    .. code-block:: txt

        some/path/000.mp4 1
        some/path/001.mp4 1
        some/path/002.mp4 2
        some/path/003.mp4 2
        some/path/004.mp4 3
        some/path/005.mp4 3


    Args:
        ann_file (str): Path to the annotation file.
        pipeline (list[dict | callable]): A sequence of data transforms.
        start_index (int): Specify a start index for frames in consideration of
            different filename format. However, when taking videos as input,
            it should be set to 0, since frames loaded from videos count
            from 0. Default: 0.
        **kwargs: Keyword arguments for ``BaseDataset``.
    """

    def __init__(
        self,
        ann_file,
        pipeline,
        start_index=0,
        scene_feature_root: str = None,
        scene_pred_root: str = None,
        **kwargs
    ):
        super().__init__(ann_file, pipeline, start_index=start_index, **kwargs)

        assert osp.exists(scene_feature_root), f"scene feature root {scene_feature_root} is not a valid path"
        assert osp.exists(scene_pred_root), f"scene prediction root {scene_pred_root} is not a valid path"
        self.scene_feature_root = scene_feature_root
        self.scene_pred_root = scene_pred_root

    def load_annotations(self):
        """Load annotation file to get video information."""
        if self.ann_file.endswith(".json"):
            return self.load_json_annotations()

        video_infos = []
        with open(self.ann_file, "r") as fin:
            for line in fin:
                line_split = line.strip().split()
                if self.multi_class:
                    assert self.num_classes is not None
                    filename, label = line_split[0], line_split[1:]
                    label = list(map(int, label))
                else:
                    filename, label = line_split
                    label = int(label)
                if self.data_prefix is not None:
                    filename = osp.join(self.data_prefix, filename)
                video_infos.append(dict(filename=filename, label=label))
        return video_infos

    def get_scene_feature_and_pred(self, results: Dict):
        # scene feature
        if "frame_dir" in results.keys():
            relative_path = results['frame_dir'].split('/')[-2:]
            relative_path = '/'.join(relative_path)
        elif "filename" in results.keys():
            if "diving48" in results['filename']:
                relative_path = results['filename'].split('/')[-2:]  # for diving48 dataset
            else:
                relative_path = results['filename'].split('/')[-3:]  # for mit dataset
            relative_path = '/'.join(relative_path)[:-4]
        else:
            raise NotImplementedError

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
