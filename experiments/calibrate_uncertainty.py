from pathlib import Path
import argparse

import numpy as np
from matplotlib import pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Calibrate uncertainty")
    parser.add_argument("base_file", type=str)
    parser.add_argument("biased_file", type=str)
    args = parser.parse_args()
    return args


def calibrate(opt):

    base_result = np.load(opt.base_file, allow_pickle=True)
    base_train_uncertainties = base_result["train_unctt"]
    base_ind_uncertainties = base_result["ind_unctt"].squeeze()  # (N1,)
    base_ood_uncertainties = base_result["ood_unctt"].squeeze()  # (N2,)
    base_ind_result = base_result["ind_pred"]  # (N1,)
    base_ood_result = base_result["ood_pred"]  # (N2,)
    base_ind_labels = base_result["ind_label"]
    base_ood_labels = base_result["ood_label"]

    biased_result = np.load(opt.biased_file, allow_pickle=True)
    biased_train_uncertainties = biased_result["train_unctt"]
    biased_ind_uncertainties = biased_result["ind_unctt"].squeeze()  # (N1,)
    biased_ood_uncertainties = biased_result["ood_unctt"].squeeze()  # (N2,)
    biased_ind_result = biased_result["ind_pred"]  # (N1,)
    biased_ood_result = biased_result["ood_pred"]  # (N2,)
    biased_ind_labels = biased_result["ind_label"]
    biased_ood_labels = biased_result["ood_label"]

    calibrate_train_uncertainties = (
        base_train_uncertainties - biased_train_uncertainties
    )
    calibrate_ind_uncertainties = base_ind_uncertainties - biased_ind_uncertainties
    calibrate_ood_uncertainties = base_ood_uncertainties - biased_ood_uncertainties
    sorted_train_uncertainty = np.sort(calibrate_train_uncertainties)[::-1]
    N = sorted_train_uncertainty.shape[0]
    topk = int(N * 0.05)
    threshold = sorted_train_uncertainty[topk]

    output_path = Path(opt.base_file).parent
    if "shuffle" in opt.biased_file:
        calibrate_suffix = "shuffle"
    elif "freeze" in opt.biased_file:
        calibrate_suffix = "freeze"
    else:
        raise NotImplementedError
    output_stem = Path(opt.base_file).stem + "_calibrate_" + calibrate_suffix + ".npz"
    output_path = (output_path / output_stem).as_posix()
    np.savez(
        output_path,
        train_unctt=calibrate_train_uncertainties,
        ind_conf=base_result["ind_conf"],
        ood_conf=base_result["ood_conf"],
        ind_unctt=calibrate_ind_uncertainties,
        ood_unctt=calibrate_ood_uncertainties,
        ind_pred=base_result["ind_pred"],
        ood_pred=base_result["ood_pred"],
        ind_label=base_result["ind_label"],
        ood_label=base_result["ood_label"],
        ind_raw_score=base_result["ind_raw_score"],
        threshold=threshold,
    )

    plt.figure(figsize=(5, 4))
    plt.hist(
        [
            calibrate_train_uncertainties,
            calibrate_ind_uncertainties,
            calibrate_ood_uncertainties,
        ],
        50,
        density=True,
        histtype="bar",
        color=["green", "blue", "red"],
        label=[
            "training set",
            "in-distribution",
            "out-of-distribution",
        ],
    )
    plt.axvline(x=threshold, color="black", label="threshold", linestyle="--")
    plt.legend()
    plt.ylabel("density")
    plt.tight_layout()
    plt.savefig(output_path[:-4] + "_distribution.png")
    plt.close()
    print(f"result file is saved to {output_path}")


if __name__ == "__main__":
    opt = parse_args()
    calibrate(opt)
