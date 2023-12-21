import os.path as osp
import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import matplotlib.pyplot as plt


def parse_args():
    """Command instruction:
    source activate mmaction
    python experiments/compare_openness.py --ind_ncls 101 --ood_ncls 51
    """
    parser = argparse.ArgumentParser(description="Compare the performance of openness")
    # model config
    parser.add_argument("result_file", type=str, nargs="+")
    parser.add_argument("--base_model", default="i3d", help="the backbone model name")
    parser.add_argument("--baselines", nargs="+", default=["AE"])
    parser.add_argument("--styles", nargs="+", default=["-b"])
    parser.add_argument(
        "--ind_ncls",
        type=int,
        default=101,
        help="the number of classes in known dataset",
    )
    parser.add_argument(
        "--ood_ncls",
        type=int,
        help="the number of classes in unknwon dataset",
        default=51,
    )
    parser.add_argument("--ood_data", default="HMDB", help="the name of OOD dataset.")
    parser.add_argument(
        "--num_rand",
        type=int,
        default=10,
        help="the number of random selection for ood classes",
    )
    args = parser.parse_args()
    return args


def main():

    opt = parse_args()
    assert len(opt.result_file) == 1, "Only support single evaluation for now!"
    plt.figure(figsize=(8, 5))  # (w, h)
    # plt.rcParams["font.family"] = "Arial"  # Times New Roman
    fontsize = 15
    assert osp.exists(
        opt.result_file[0]
    ), f"File {opt.result_file[0]} not found! Run ood_detection first!"
    # load the testing results
    results = np.load(opt.result_file[0], allow_pickle=True)
    train_uncertainties = results["train_unctt"]
    ind_uncertainties = results["ind_unctt"].squeeze()  # (N1,)
    ood_uncertainties = results["ood_unctt"].squeeze()  # (N2,)
    ind_results = results["ind_pred"]  # (N1,)
    ood_results = results["ood_pred"]  # (N2,)
    ind_labels = results["ind_label"]
    ood_labels = results["ood_label"]
    threshold = results["threshold"]
    # get threshold
    # sorted_train_uncertainty = np.sort(train_uncertainties)[::-1]
    # N = sorted_train_uncertainty.shape[0]
    # topk = N - int(N * 0.95)
    # threshold = sorted_train_uncertainty[topk - 1]

    # close-set accuracy (multi-class)
    acc = accuracy_score(ind_labels, ind_results)
    # open-set auc-roc (binary class)
    preds = np.concatenate((ind_results, ood_results), axis=0)
    uncertains = np.concatenate((ind_uncertainties, ood_uncertainties), axis=0)
    preds[uncertains > threshold] = 1
    preds[uncertains <= threshold] = 0
    labels = np.concatenate((np.zeros_like(ind_labels), np.ones_like(ood_labels)))
    aupr = roc_auc_score(labels, preds)
    print(
        "Model: {}, Closed-Set Accuracy (multi-class): {:.3f}, Open-Set AUC (bin-class): {:.3f}, threshold: {}".format(
            opt.baselines, acc * 100, aupr * 100, threshold
        )
    )

    # open set F1 score (multi-class)
    # falsely rejection
    ind_results[ind_uncertainties > threshold] = opt.ind_ncls
    macro_F1_list = [f1_score(ind_labels, ind_results, average="macro")]
    std_list = [0]
    openness_list = [0]
    for n in range(opt.ood_ncls):
        ncls_novel = n + 1
        openness = (
            1 - np.sqrt((2 * opt.ind_ncls) / (2 * opt.ind_ncls + ncls_novel))
        ) * 100
        openness_list.append(openness)
        # randoml select the subset of ood samples
        macro_F1_multi = np.zeros((opt.num_rand), dtype=np.float32)
        for m in range(opt.num_rand):
            cls_select = np.random.choice(opt.ood_ncls, ncls_novel, replace=False)
            ood_sub_results = np.concatenate(
                [ood_results[ood_labels == clsid] for clsid in cls_select]
            )
            ood_sub_uncertainties = np.concatenate(
                [ood_uncertainties[ood_labels == clsid] for clsid in cls_select]
            )
            ood_sub_results[
                ood_sub_uncertainties > threshold
            ] = opt.ind_ncls  # correctly rejection
            ood_sub_labels = np.ones_like(ood_sub_results) * opt.ind_ncls
            # construct preds and labels
            preds = np.concatenate((ind_results, ood_sub_results), axis=0)
            labels = np.concatenate((ind_labels, ood_sub_labels), axis=0)
            macro_F1_multi[m] = f1_score(labels, preds, average="macro")
        macro_F1 = np.mean(macro_F1_multi)
        std = np.std(macro_F1_multi)
        macro_F1_list.append(macro_F1)
        std_list.append(std)

    # draw comparison curves
    macro_F1_list = np.array(macro_F1_list)
    std_list = np.array(std_list)
    plt.plot(openness_list, macro_F1_list * 100, opt.styles, linewidth=2)
    # plt.fill_between(openness_list, macro_F1_list - std_list, macro_F1_list + std_list, style)

    w_openness = np.array(openness_list) / 100.0
    open_maF1_mean = np.sum(w_openness * macro_F1_list) / np.sum(w_openness)
    open_maF1_std = np.sum(w_openness * std_list) / np.sum(w_openness)
    print(
        "Open macro-F1 score: %.3f, std=%.3lf"
        % (open_maF1_mean * 100, open_maF1_std * 100)
    )

    plt.xlim(0, max(openness_list))
    plt.ylim(40, 100)
    plt.xlabel("Openness (%)", fontsize=fontsize)
    plt.ylabel("macro F1 (%)", fontsize=fontsize)
    plt.grid("on")
    plt.legend(opt.baselines)
    # plt.legend(['MC Dropout BALD', 'BNN SVI BALD', 'DEAR (vanilla)',
    #            'DEAR (alter)', 'DEAR (joint)'], loc='lower left', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.tight_layout()
    result_path = Path(opt.result_file[0]).parent
    png_file_name = Path(opt.result_file[0]).name.replace("npz", "png")
    png_file = (result_path / png_file_name).as_posix()
    plt.title(
        "Closed-set acc: {:.3f}%; open-set AUC: {:.3f}%; open mF1 score: {:.3f}+-{:.3f}".format(
            acc * 100, aupr * 100, open_maF1_mean * 100, open_maF1_std * 100
        )
    )
    plt.tight_layout()
    plt.savefig(png_file)
    # plt.savefig(png_file.replace("png", "pdf"))
    print("Openness curve figure is saved in: %s" % (png_file))
    plt.close()


if __name__ == "__main__":

    main()
