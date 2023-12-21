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

ucf_to_hmdb_class_index_mapper = {
    73: 5,
    74: 5,
    27: 13,
    32: 15,
    84: 21,
    69: 26,
    16: 27,
    17: 27,
    70: 27,
    71: 29,
    10: 30,
    41: 31,
    7: 34,
    2: 35,
    97: 49,
}


def parse_args():
    parser = argparse.ArgumentParser(description="open-set performance evaluation")
    # model config
    parser.add_argument("result_file", type=str)
    parser.add_argument(
        "--num_ind_class",
        type=int,
        default=101,
        help="the number of classes in the known dataset",
    )
    parser.add_argument(
        "--ood_datalist",
        default="data/hmdb51/hmdb51_val_split_1_videos.txt",
        type=str,
    )
    parser.add_argument(
        "--num_rand",
        type=int,
        default=10,
        help="the number of random selection for ood classes",
    )
    parser.add_argument("-tp", "--threshold_portion", type=float, default=None)
    parser.add_argument("-t", "--threshold", type=float, default=None)
    parser.add_argument("-v", "--verbose", default=False, action="store_true")
    parser.add_argument("--suffix", type=str, default="")
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


def get_ood_acc(results, labels) -> float:
    for i in range(results.shape[0]):
        if results[i] in ucf_to_hmdb_class_index_mapper.keys():
            results[i] = ucf_to_hmdb_class_index_mapper[results[i]]
        else:
            results[i] = 0
    acc = accuracy_score(results, labels)
    return acc


def main(
    opt,
    default_hmdb51_datalist: str = "data/hmdb51/hmdb51_val_split_1_videos.txt",
    default_mit_datalist: str = "data/mit/mit_val_list_videos.txt",
    clean_hmdb51_datalist: str = "data/hmdb51/hmdb51_val_split_1_videos_clean.txt",
):
    # assertions
    if opt.threshold is not None:
        assert (
            opt.threshold_portion is None
        ), "only one of threshold or threshold_portion can not be None"
    if opt.threshold_portion is not None:
        assert 1 >= opt.threshold_portion >= 0
        assert (
            opt.threshold is None
        ), "only one of threshold or threshold_portion can not be None"
    assert "video" in opt.ood_datalist, "only support video datalist"
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
    print(f"The threshold to compute open maF1 is set to {threshold}")

    # get class names
    class_names = get_class_names(opt.ood_datalist)
    num_ood_class = len(class_names)
    if "hmdb" in opt.result_file.lower():
        ood_overlap_indices = get_indices(
            clean_hmdb51_datalist, ood_video_names.tolist(), select_non_exist=True
        )
        ood_generalization_acc = get_ood_acc(
            ood_results[ood_overlap_indices],
            ood_labels[ood_overlap_indices],
        )
    else:
        ood_generalization_acc = -1

    # select results from videos in the datalist, this might remove several
    # videos according to the ood_datalist
    ood_indices = get_indices(opt.ood_datalist, ood_video_names.tolist())
    ood_uncertainties = ood_uncertainties[ood_indices]
    ood_results = ood_results[ood_indices]
    ood_labels = ood_labels[ood_indices]
    ood_video_names = ood_video_names[ood_indices]

    # close-set accuracy (multi-class)
    acc = accuracy_score(ind_labels, ind_results)

    # open-set AUROC (binary class)
    preds = np.concatenate((ind_results, ood_results), axis=0)
    uncertains = np.concatenate((ind_uncertainties, ood_uncertainties), axis=0)
    preds[uncertains > threshold] = 1
    preds[uncertains <= threshold] = 0
    labels = np.concatenate((np.zeros_like(ind_labels), np.ones_like(ood_labels)))
    hard_auroc = roc_auc_score(labels, preds)
    auroc = roc_auc_score(labels, uncertains)
    fpr, tpr, _ = roc_curve(labels, uncertains)
    fnr = 1 - tpr
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    fpr95_idx = np.abs(tpr - 0.95).argmin()
    fpr95 = fpr[fpr95_idx]
    tpr10_idx = np.abs(fpr - 0.1).argmin()
    tpr10 = tpr[tpr10_idx]

    result = {
        "overall open set auc": auroc,
        "fpr@95tpr": fpr95,
        "tpr@10fpr": tpr10,
        "hard overall open set auc": hard_auroc,
        "closed set acc": acc,
        "threshold": threshold,
        "per class open set auc": {},
        "ood generalization acc": ood_generalization_acc,
    }

    # per-class AUROC
    for i in range(len(class_names)):
        indices = np.where(ood_labels == i)
        uncertainties = np.concatenate(
            (ind_uncertainties, ood_uncertainties[indices]), axis=0
        )
        predicted_ind_indices = np.where(uncertainties > threshold)
        predicted_ood_indices = np.where(uncertainties <= threshold)
        pred = np.zeros_like(uncertainties)
        pred[predicted_ood_indices] = 1
        labels = np.concatenate(
            (np.zeros_like(ind_uncertainties), np.ones_like(ood_uncertainties[indices]))
        )
        try:
            auroc_per_class = roc_auc_score(labels, uncertainties)
        except Exception as e:
            if opt.verbose:
                print(
                    f"failed to compute AUROC for class '{class_names[i]}' due to: {e}"
                )
            continue

        if opt.verbose:
            print(f"AUROC for class '{class_names[i]}' is {auroc_per_class}")
        result["per class open set auc"][class_names[i]] = auroc_per_class

    # f1s
    tmp_ood_labels = np.ones_like(ood_results) * opt.num_ind_class
    labels = np.concatenate([ind_labels, tmp_ood_labels], axis=0)
    ind_results[ind_uncertainties > threshold] = opt.num_ind_class  # false rejection
    ood_results[ood_uncertainties > threshold] = opt.num_ind_class
    preds = np.concatenate([ind_results, ood_results], axis=0)
    macro_f1 = f1_score(labels, preds, average="macro")
    micro_f1 = f1_score(labels, preds, average="micro")
    result["marco f1"] = macro_f1
    result["micro f1"] = micro_f1

    # open macro F1
    ind_results[ind_uncertainties > threshold] = opt.num_ind_class  # false rejection
    macro_F1_list = [f1_score(ind_labels, ind_results, average="macro")]
    std_list = [0]
    openness_list = [0]
    for n in range(num_ood_class):
        ncls_novel = n + 1
        openness = (
            1 - np.sqrt((2 * opt.num_ind_class) / (2 * opt.num_ind_class + ncls_novel))
        ) * 100
        openness_list.append(openness)
        # randoml select the subset of ood samples
        macro_F1_multi = np.zeros((opt.num_rand), dtype=np.float32)
        for m in range(opt.num_rand):
            cls_select = np.random.choice(num_ood_class, ncls_novel, replace=False)
            ood_sub_results = np.concatenate(
                [ood_results[ood_labels == clsid] for clsid in cls_select]
            )
            ood_sub_uncertainties = np.concatenate(
                [ood_uncertainties[ood_labels == clsid] for clsid in cls_select]
            )
            # correct rejection
            ood_sub_results[ood_sub_uncertainties > threshold] = opt.num_ind_class
            ood_sub_labels = np.ones_like(ood_sub_results) * opt.num_ind_class
            # construct preds and labels
            preds = np.concatenate((ind_results, ood_sub_results), axis=0)
            labels = np.concatenate((ind_labels, ood_sub_labels), axis=0)
            macro_F1_multi[m] = f1_score(labels, preds, average="macro")
        macro_F1 = np.mean(macro_F1_multi)
        std = np.std(macro_F1_multi)
        macro_F1_list.append(macro_F1)
        std_list.append(std)
    macro_F1_list = np.array(macro_F1_list)
    std_list = np.array(std_list)
    w_openness = np.array(openness_list) / 100.0
    open_maF1_mean = np.sum(w_openness * macro_F1_list) / np.sum(w_openness)
    open_maF1_std = np.sum(w_openness * std_list) / np.sum(w_openness)
    result[
        "open macro f1"
    ] = f"{open_maF1_mean * 100:.3f} +- {open_maF1_std * 100:.3f}%"

    # save and print results
    with open(opt.result_file.replace(".npz", f"{opt.suffix}.json"), "w") as f:
        json.dump(result, f)

    print(
        f"Closed-Set Accuracy (multi-class): {acc * 100:.3f}, "
        f"Open-Set AUC (bin-class): {auroc * 100:.3f}, "
        f"open macro f1: {result['open macro f1']}, "
        f"macro F1: {macro_f1 * 100:.3f}%, "
        f"micro F1: {micro_f1 * 100:.3f}%, "
        f"FPR@95TPR: {fpr95 * 100:.3f}, TPR@10FPR: {tpr10 * 100:.3f}, "
        f"Open-Set AUC (hard): {hard_auroc * 100:.3f}, threshold: {threshold} "
        f"OOD Generalization Accuracy: {ood_generalization_acc * 100:.3f}"
    )

    # save figure
    plt.plot(fpr, tpr)
    plt.plot(np.arange(0, 1.1, 0.1), np.arange(0, 1.1, 0.1), "k")
    plt.ylabel("True positive rate")
    plt.xlabel("False positive rate")
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.title(
        f"ROC curve: EER {eer * 100:.3f}%, AUC {auroc * 100:.3f}%, "
        f"FPR@95 {fpr95 * 100: .3f}%, TPR@10 {tpr10 * 100: .3f}, "
        f"hard AUC {hard_auroc * 100: .3f}%",
        loc="center",
        wrap=True,
    )
    plt.gca().set_aspect("equal")
    plt.tight_layout()
    plt.savefig(opt.result_file.replace(".npz", f"_roc{opt.suffix}.png"), dpi=400)
    print(f"figure saved to {opt.result_file.replace('.npz', f'_roc{opt.suffix}.png')}")


if __name__ == "__main__":
    opt = parse_args()
    main(opt)
