from typing import List, Set
import os.path as osp
import argparse
from pathlib import Path
import json
import time

from rich import print
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, roc_curve
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="open-set performance evaluation")
    # model config
    parser.add_argument("result_file", type=str)
    parser.add_argument(
        "--ood_datalist",
        default="data/hmdb51/hmdb51_val_split_1_videos.txt",
        type=str,
    )
    parser.add_argument("-tp", "--threshold_portion", type=float, default=None)
    parser.add_argument("-t", "--threshold", type=float, default=None)
    parser.add_argument("--clean", action="store_true", default=False)
    args = parser.parse_args()
    return args


def get_class_names(datalist: str) -> Set[str]:
    class_names = set()
    datalist = Path(opt.ood_datalist)
    if "hmdb" in datalist.as_posix().lower():
        anno_dir = datalist.parent / "annotations"
        class_names = set(
            "_".join(x.stem.split("_")[:-2]) for x in anno_dir.glob("*test_split1.txt")
        )
    elif "mit" in datalist.as_posix().lower():
        anno_file = datalist.parent / "annotations" / "moments_categories.txt"
        with open(anno_file, "r") as f:
            for line in f:
                line = line.strip().split(",")
                class_name = line[0]
                class_names.add(class_name)
    else:
        raise NotImplementedError(f"not supported datalist {datalist.lower()}")

    class_names: Set[str] = sorted(class_names)
    return class_names


def get_indices(
    datalist: str, video_names: List, select_non_exist: bool = False
) -> List:
    video_names_in_datalist = []
    with open(datalist, "r") as f:
        for line in f:
            line = line.strip().split(" ")
            video_name = line[0].split("/")[-1]
            video_names_in_datalist.append(video_name)
    selected_indices = []
    for i, video_name in enumerate(video_names):
        if (not select_non_exist) and (video_name in video_names_in_datalist):
            selected_indices.append(i)
        elif select_non_exist and (video_name not in video_names_in_datalist):
            selected_indices.append(i)
    if len(selected_indices) != len(video_names):
        print(
            f"[red]{len(video_names) - len(selected_indices)} OOD videos are "
            "excluded from the ood detection results[/red]"
        )
        time.sleep(1)
    return selected_indices


def main(
    opt,
    default_hmdb51_datalist: str = "data/hmdb51/hmdb51_val_split_1_videos.txt",
    default_mit_datalist: str = "data/mit/mit_val_list_videos.txt",
    clean_hmdb51_datalist: str = "data/hmdb51/hmdb51_val_split_1_videos_clean.txt",
):
    assert osp.exists(
        opt.result_file
    ), f"File {opt.result_file} not found! Run ood_detection first!"
    if "hmdb" in opt.result_file.lower():
        if "hmdb" not in opt.ood_datalist.lower():
            print(
                f"[red]result file {opt.result_file} and ood datalist "
                f"{opt.ood_datalist} do not match[/red]"
            )
            opt.ood_datalist = default_hmdb51_datalist
            print(f"[red]changing ood datalist to {opt.ood_datalist}[/red]")
    elif "mit" in opt.result_file.lower():
        if "mit" not in opt.ood_datalist.lower():
            print(
                f"[red]result file {opt.result_file} and ood datalist "
                f"{opt.ood_datalist} do not match[/red]"
            )
            opt.ood_datalist = default_mit_datalist
            print(f"[red]changing ood datalist to {opt.ood_datalist}[/red]")
    else:
        raise NotImplementedError(
            f"no dataset (hmdb or mit) detected in {opt.result_file}"
        )

    if opt.clean:
        assert (
            "hmdb" in opt.result_file.lower()
        ), "only hmdb51 dataset supports clean evaluation"
        opt.ood_datalist = clean_hmdb51_datalist
        opt.suffix = opt.suffix + "_clean"

    # load the testing results
    results = np.load(opt.result_file, allow_pickle=True)
    train_uncertainties = results["train_unctt"].squeeze()  # num_train
    ind_uncertainties = results["ind_unctt"].squeeze()  # num_ind_test
    ood_uncertainties = results["ood_unctt"].squeeze()  # num_ood_test
    ind_results = results["ind_pred"]  # num_ind_test
    ood_results = results["ood_pred"]  # num_ood_test
    ind_labels = results["ind_label"]  # num_ind_test
    ood_labels = results["ood_label"]  # num_ind_test
    threshold = results["threshold"]  # 1
    ood_video_names = results["ood_video_names"]  # num_ind_test

    sorted_train_uncertainty = np.sort(train_uncertainties)[::-1]
    N = sorted_train_uncertainty.shape[0]
    if opt.threshold_portion is not None:
        topk = min(N - 1, max(0, int(N * (1 - opt.threshold_portion))))
        threshold = float(sorted_train_uncertainty[topk])
    elif opt.threshold is not None:
        threshold = opt.threshold
    else:
        topk = min(N - 1, max(0, int(N * 0.05)))
        threshold = float(sorted_train_uncertainty[topk])

    # select results from videos in the datalist, this might remove several
    # videos according to the ood_datalist
    ood_indices = get_indices(opt.ood_datalist, ood_video_names.tolist())
    ood_uncertainties = ood_uncertainties[ood_indices]
    ood_results = ood_results[ood_indices]
    ood_labels = ood_labels[ood_indices]
    ood_video_names = ood_video_names[ood_indices]

    ind_uncertainties = np.array(ind_uncertainties)
    ind_uncertainties = (ind_uncertainties - np.min(ind_uncertainties)) / (
        np.max(ind_uncertainties) - np.min(ind_uncertainties)
    )  # normalize
    ood_uncertainties = np.array(ood_uncertainties)
    ood_uncertainties = (ood_uncertainties - np.min(ood_uncertainties)) / (
        np.max(ood_uncertainties) - np.min(ood_uncertainties)
    )  # normalize

    plt.style.use(["science", "grid"])
    plt.figure(figsize=(5, 4))
    plt.hist(
        [ind_uncertainties, ood_uncertainties],
        50,
        density=True,
        histtype="bar",
        color=["blue", "red"],
        label=[
            "in-distribution (UCF101)",
            "out-of-distribution (MiTv2)",
        ],
    )
    plt.xlabel("EDL Uncertainty")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(opt.result_file.replace(".npz", "_distribution.png"), dpi=300)
    plt.savefig(opt.result_file.replace(".npz", "_distribution.pdf"), dpi=300)


if __name__ == "__main__":
    opt = parse_args()
    main(opt)
