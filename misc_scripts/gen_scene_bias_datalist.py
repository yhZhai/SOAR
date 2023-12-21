from typing import Optional, List
from pathlib import Path
import json

from tqdm import tqdm
import numpy as np
from scipy.spatial import distance


class VideoItem:
    def __init__(self, video_name: str, video_path: str, class_idx: int, scene_feature):
        self.video_name = video_name
        self.video_path = video_path
        self.class_idx = class_idx
        self.scene_feature = scene_feature
        self.distance_to_train_set = None
        self.most_similar_video_path = None
        self.least_similar_video_path = None
        self.maximal_distance_to_train_set = None


def compute_distance(
    train_scene_feature_list: List,
    train_video_path_list: List[str],
    ood_video_item_list: List[VideoItem],
):
    def _single_compute_distance(ood_video_item: VideoItem):
        distance_list = []
        for train_scene_feature in train_scene_feature_list:
            cosine_distance = distance.cosine(
                train_scene_feature, ood_video_item.scene_feature
            )
            distance_list.append(cosine_distance)
        ood_video_item.distance_to_train_set = min(distance_list)
        ood_video_item.maximal_distance_to_train_set = max(distance_list)
        ood_video_item.most_similar_video_path = train_video_path_list[
            np.argmin(distance_list)
        ]
        ood_video_item.least_similar_video_path = train_video_path_list[
            np.argmax(distance_list)
        ]

    for item in ood_video_item_list:
        _single_compute_distance(item)


def main(
    train_datalist: str,
    ood_datalist: str,
    train_scene_root: str,
    ood_scene_root: str,
    num_split: int,
    output_template: str,
    ood_num_class: int,
    meta_save_path: str,
):
    train_datalist = Path(train_datalist)
    ood_datalist = Path(ood_datalist)
    train_scene_root = Path(train_scene_root)
    ood_scene_root = Path(ood_scene_root)

    # load training dataset
    train_scene_feature_list = []
    train_video_path_list = []
    with open(train_datalist.as_posix(), "r") as f:
        for i, line in tqdm(enumerate(f), desc="loading training dataset"):
            # # for debug purpose
            # if i > 10:
            #     break
            line = line.strip().split(" ")[0]
            train_video_path_list.append(line)
            line = line.replace(".avi", ".npy")
            scene_feature_path = train_scene_root / Path(line)
            scene_feature = np.load(scene_feature_path.as_posix()).squeeze(0)
            train_scene_feature_list.append(scene_feature)

    # load ood dataset
    ood_video_list = [[] for _ in range(ood_num_class)]
    with open(ood_datalist.as_posix(), "r") as f:
        for i, line in tqdm(enumerate(f), desc="loading ood dataset"):
            line = line.strip().split(" ")
            video_path = Path(line[0])
            video_name = video_path.name
            class_idx = int(line[1])
            scene_feature_path = ood_scene_root / video_path
            suffix = scene_feature_path.suffix
            scene_feature = np.load(
                scene_feature_path.as_posix().replace(suffix, ".npy")
            ).squeeze(0)
            ood_video_list[class_idx].append(
                VideoItem(video_name, video_path.as_posix(), class_idx, scene_feature)
            )

    # compute distance
    for i in tqdm(range(len(ood_video_list)), desc="computing cosine distance"):
        compute_distance(
            train_scene_feature_list, train_video_path_list, ood_video_list[i]
        )
        ood_video_list[i] = sorted(
            ood_video_list[i], key=lambda x: x.distance_to_train_set
        )

    # generate ood dataset splits
    ood_dataset_splits = [[] for _ in range(num_split)]
    avg_distance = [[] for _ in range(num_split)]
    for cls_idx, ood_video in enumerate(ood_video_list):
        for i, item in enumerate(ood_video):
            split_idx = int(
                min(max(0, i / (len(ood_video) / num_split)), num_split - 1)
            )
            ood_dataset_splits[split_idx].append(item)
            avg_distance[split_idx].append(item.distance_to_train_set)
    avg_distance = [sum(dis) / len(dis) for dis in avg_distance]
    print("Number of videos in each split:", list(map(len, ood_dataset_splits)))
    print("Avgerage cosine distance of each split:", avg_distance)

    for i in range(num_split):
        with open(output_template.format(i, avg_distance[i]), "w") as f:
            for item in ood_dataset_splits[i]:
                f.write("{} {}\n".format(item.video_path, item.class_idx))

    metadata = {}
    for cls_items in ood_video_list:
        for item in cls_items:
            metadata[item.video_path] = {
                "most similar video path": item.most_similar_video_path,
                "least similar video path": item.least_similar_video_path,
                "minimal distance": item.distance_to_train_set,
                "maximal distance": item.maximal_distance_to_train_set,
            }
    with open(meta_save_path, "w") as f:
        json.dump(metadata, f)


if __name__ == "__main__":
    # # hmdb
    # main(
    #     train_datalist="data/ucf101/ucf101_train_split_1_videos.txt",
    #     ood_datalist="data/hmdb51/hmdb51_val_split_1_videos.txt",
    #     train_scene_root="data/ucf101_scene_feature",
    #     ood_scene_root="data/hmdb51_scene_feature",
    #     num_split=10,
    #     output_template="hmdb51_val_split_1_scene_bias_analysis_{}_dis{:.3f}_videos.txt",
    #     ood_num_class=51,
    #     meta_save_path="hmdb_scene_bias_metadata.json",
    # )

    # mit
    main(
        train_datalist="data/ucf101/ucf101_train_split_1_videos.txt",
        ood_datalist="data/mit/mit_val_list_videos.txt",
        train_scene_root="data/ucf101_scene_feature",
        ood_scene_root="data/mit_scene_feature",
        num_split=15,
        output_template="mit_val_scene_bias_analysis_{}_dis{:.3f}_videos.txt",
        ood_num_class=305,
        meta_save_path="mit_scene_bias_metadata.json",
    )

    # # ucf101
    # main(
    #     train_datalist="data/ucf101/ucf101_train_split_1_videos.txt",
    #     ood_datalist="data/ucf101/ucf101_val_split_1_videos.txt",
    #     train_scene_root="data/ucf101_scene_feature",
    #     ood_scene_root="data/ucf101_scene_feature",
    #     num_split=20,
    #     output_template="ucf101_val_split_1_scene_bias_analysis_{}_dis{:.3f}_videos.txt",
    #     ood_num_class=101,
    #     meta_save_path="ucf101_scene_bias_metadata.json",
    # )
