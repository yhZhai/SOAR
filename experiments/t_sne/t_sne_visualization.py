"""
t-SNE visualization usage:
1. python experiments/t_sne_generation.py {config_path} {checkpoint_path}
    --out {output_json_path}
2. python experiments/t_sne_visualization.py -src {output_json_path} -out
    {output_figure_path} -title {figure_title}
"""

import os
import json
import argparse

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns


def main(opt):
    assert os.path.exists(opt.src)
    with open(opt.src, "r") as f:
        file = json.load(f)

    x, labels = [], []
    for k, v in file.items():
        x.append(v["feat"])
        labels.append(v["label"])
    x = np.asarray(x, dtype="float64")
    labels = np.asarray(labels, dtype="int32")

    x_embedded = TSNE(n_components=2, init="random").fit_transform(x)

    df = pd.DataFrame()
    df["x"] = x_embedded[:, 0]
    df["y"] = x_embedded[:, 1]
    df["labels"] = labels

    plt.figure(figsize=(16, 16))
    sns.scatterplot(
        x="x",
        y="y",
        hue="labels",
        palette=sns.color_palette("hls", 101),
        data=df,
        legend="full",
        alpha=0.3,
    ).set(title=opt.title)
    plt.savefig(opt.out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("t-SNE")
    parser.add_argument("-src", type=str)
    parser.add_argument("-out", type=str)
    parser.add_argument("-title", type=str)
    opt = parser.parse_args()
    main(opt)
