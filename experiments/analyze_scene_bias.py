import os.path as osp
import argparse
from pathlib import Path
import tqdm
from typing import Optional, Dict, List, Tuple
import json
import time
import re

from rich import print
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from scipy.stats import pearsonr
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze scene bias")
    parser.add_argument("result_file", type=str)
    parser.add_argument(
        "-d", "--dataset", type=str, default="hmdb51", choices=["hmdb51", "mit"]
    )
    parser.add_argument(
        "--datalist_tmpl",
        type=str,
        default="data/{}/{}_val*scene_bias_analysis_{}_dis*videos.txt",
    )
    parser.add_argument(
        "-as", "--alter_set", type=str, default="open", choices=["open", "close"]
    )
    parser.add_argument("--clean", action="store_true", default=False)
    parser.add_argument("-tp", "--threshold_portion", type=float, default=None)
    parser.add_argument("-t", "--threshold", type=float, default=None)
    args = parser.parse_args()
    return args


def get_video_name_from_datalist(
    datalist: str, filter_video_name_list: Optional[List[str]] = None
) -> List[str]:
    assert osp.exists(datalist)
    result = []
    with open(datalist, "r") as f:
        for line in f:
            line = line.strip()
            video_path = line.split(" ")[0]
            video_name = Path(video_path).name
            if filter_video_name_list is None:
                result.append(video_name)
            else:
                if video_name in filter_video_name_list:
                    result.append(video_name)

    return result


def get_ood_datalist(dataset: str, datalist_tmpl: str) -> Tuple[List[str], List[float]]:
    idx = 0
    datalist_list = []
    distance_list = []
    while True:
        datalist = datalist_tmpl.format(dataset, dataset, idx)
        glob_result = list(Path().glob(datalist))
        if len(glob_result) == 1:
            datalist = glob_result[0].as_posix()
            datalist_list.append(datalist)
            distance = re.search(r"_dis[0-9.]+_", datalist).group()
            distance = float(distance[4:-1])
            distance_list.append(distance)
        elif len(glob_result) == 0:
            break
        else:
            raise Exception(
                f"found {len(glob_result)} matches for {datalist}: {glob_result}"
            )
        idx += 1

    return datalist_list, distance_list


def get_open_maf1(
        num_ood_class: int,
        num_ind_class: int,
        num_rand: int,
        ind_results,
        ind_labels,
        ood_results,
        ood_labels,
        ood_uncertainties,
        threshold,
):
    macro_F1_list = [f1_score(ind_labels, ind_results, average="macro")]
    std_list = [0]
    openness_list = [0]
    for n in range(num_ood_class):
        ncls_novel = n + 1
        openness = (
            1 - np.sqrt((2 * num_ind_class) / (2 * num_ind_class + ncls_novel))
        ) * 100
        openness_list.append(openness)
        # randoml select the subset of ood samples
        macro_F1_multi = np.zeros((num_rand), dtype=np.float32)
        for m in range(num_rand):
            cls_select = np.random.choice(num_ood_class, ncls_novel, replace=False)
            ood_sub_results = np.concatenate(
                [ood_results[ood_labels == clsid] for clsid in cls_select]
            )
            ood_sub_uncertainties = np.concatenate(
                [ood_uncertainties[ood_labels == clsid] for clsid in cls_select]
            )
            # correct rejection
            ood_sub_results[ood_sub_uncertainties > threshold] = num_ind_class
            ood_sub_labels = np.ones_like(ood_sub_results) * num_ind_class
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

    return open_maF1_mean, open_maF1_std


def main(
    opt,
    default_clean_hmdb51_datalist: str = "data/hmdb51/hmdb51_val_split_1_videos_clean.txt",
):

    # assertions
    assert osp.exists(
        opt.result_file
    ), f"File {opt.result_file} not found! Run ood_detection first!"
    if opt.clean:
        assert (
            "hmdb" in opt.result_file.lower()
        ), "only hmdb dataset supports clean evaluation"
        clean_video_name_list = get_video_name_from_datalist(
            default_clean_hmdb51_datalist
        )
        suffix = "_clean"
        print("running evaluation on the clean sublist")
    else:
        clean_video_name_list = None
        suffix = ""

    if opt.alter_set == "close":
        suffix = suffix + "_close"

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
    ind_video_names = results["ind_video_names"]

    if opt.alter_set == "open":
        ood_datalists, datalist_dis = get_ood_datalist(opt.dataset, opt.datalist_tmpl)
    else:
        ood_datalists, datalist_dis = get_ood_datalist("ucf101", opt.datalist_tmpl)

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

    # open-set AUROC (binary class)
    aurocs = []
    for datalist in tqdm.tqdm(ood_datalists):
        subset_ood_uncertainties = []
        subset_ind_uncertainties = []
        subset_ind_results = []
        subset_ind_labels = []
        subset_ood_results = []
        subset_ood_labels = []
        subset_video_names = get_video_name_from_datalist(
            datalist, clean_video_name_list
        )
        if opt.alter_set == "open":
            for ood_video_name, uncertainty in zip(ood_video_names, ood_uncertainties):
                if ood_video_name in subset_video_names:
                    subset_ood_uncertainties.append(uncertainty)

            subset_ood_uncertainties = np.array(subset_ood_uncertainties)

            uncertains = np.concatenate(
                (ind_uncertainties, subset_ood_uncertainties), axis=0
            )
            labels = np.concatenate(
                (np.zeros_like(ind_labels), np.ones_like(subset_ood_uncertainties))
            )
        else:  # close
            for ood_video_name, uncertainty in zip(ind_video_names, ind_uncertainties):
                if ood_video_name in subset_video_names:
                    subset_ood_uncertainties.append(uncertainty)
            subset_ind_uncertainties = np.array(subset_ood_uncertainties)

            uncertains = np.concatenate(
                (subset_ind_uncertainties, ood_uncertainties), axis=0
            )
            labels = np.concatenate(
                (
                    np.zeros_like(subset_ind_uncertainties),
                    np.ones_like(ood_uncertainties),
                )
            )

        auroc = roc_auc_score(labels, uncertains)
        aurocs.append(auroc)

    datalist_dis = np.array(datalist_dis)
    aurocs = np.array(aurocs)
    print("scene distances:", datalist_dis)
    print("subset performances:", aurocs)
    # pearson correlation
    pcc = pearsonr(datalist_dis, aurocs)
    print("===")
    print(pcc)
    # covariance
    cov_matrix = np.cov(datalist_dis, aurocs * 100)
    print("===")
    print("covariance matrix:", cov_matrix)
    print("covariance:", cov_matrix[0, 1])
    # variance
    variance = aurocs.var()
    print("===")
    print("variance:", variance)
    # linear fit
    linear_fit_coe = np.polyfit(datalist_dis, aurocs * 100, 1)
    print("===")
    print(f"polynomial coefficients: {linear_fit_coe[0]}x + {linear_fit_coe[1]}")
    result = {
        "pcc": pcc[0],
        "subset results aucs": aurocs.tolist(),
        "feature cosine distances": datalist_dis.tolist(),
        "subsets": ood_datalists,
        "covariance": cov_matrix[0, 1],
        "variance": variance,
        "linear slope": linear_fit_coe[0],
    }

    with open(opt.result_file.replace(".npz", f"_scene_bias{suffix}.json"), "w") as f:
        json.dump(result, f)
    print(
        "\nfile saved to {}".format(
            opt.result_file.replace(".npz", f"_scene_bias{suffix}.json")
        )
    )

    plt.plot(datalist_dis, aurocs, "s-")
    plt.xlabel("feature distance")
    plt.ylabel("AUROC")
    plt.grid()
    plt.title(
        f"PCC: {pcc[0]:.5f}, cov: {cov_matrix[0, 1]:.5f}, var: {variance :.5f}, slope: {linear_fit_coe[0]:.5f}"
    )
    plt.savefig(opt.result_file.replace(".npz", f"_scene_bias{suffix}.png"), dpi=300)
    plt.close()


if __name__ == "__main__":
    opt = parse_args()
    main(opt)
