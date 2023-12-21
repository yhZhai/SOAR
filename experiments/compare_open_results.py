from pathlib import Path
import json
import argparse

import numpy as np
from matplotlib import pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Openness performance comparison.")
    parser.add_argument("result_files", type=str, nargs="+")
    parser.add_argument("-n", "--result_names", type=str, nargs="+")
    parser.add_argument("-s", "--save", type=str, default="tmp/openness_comparison.png")
    opt = parser.parse_args()
    return opt


def main():
    opt = parse_args()
    assert len(opt.result_names) == 2, "only support 2 files comparison for now"  # TODO
    assert len(opt.result_names) == len(opt.result_files)
    for file in opt.result_files:
        assert Path(file).exists(), f"{file} does no exist"

    # load data
    class_names = []
    results = {}
    for result_name, result_file in zip(opt.result_names, opt.result_files):
        with open(result_file, "r") as f:
            result: dict = json.load(f)
        overall_result = result.pop("overall")
        results[result_name] = {"overall": overall_result, "detail": result}
        if not class_names:
            class_names = sorted(result.keys())

    # plot
    x = np.arange(len(class_names))
    width = 0.35
    for i, (k, result) in enumerate(results.items()):
        if i == 0:
            plt.bar(
                x,
                result["detail"].values(),
                width,
                label="{}:{:.2f}%".format(k, result["overall"] * 100),
            )
        else:
            plt.bar(
                x + width,
                result["detail"].values(),
                width,
                label="{}:{:.2f}%".format(k, result["overall"] * 100),
            )
    plt.legend(loc="lower center")
    plt.ylim(0, 1)
    plt.xticks(x + width, class_names, fontsize=8, rotation=270)
    # plt.annotate("{}: {}\n{}: {}".format(opt.result_names[0], opt.result_files[0], opt.result_names[1], opt.result_files[1]), xy=(0, 0))
    plt.tight_layout()
    plt.savefig(opt.save, dpi=400)

    # compute diff
    ## assume ours is the second file
    diff = {}
    for k in results[opt.result_names[1]]["detail"].keys():
        diff[k] = (
            results[opt.result_names[1]]["detail"][k]
            - results[opt.result_names[0]]["detail"][k]
        ) * 100
    diff = dict(sorted(diff.items(), key=lambda item: item[1]))
    print(diff)


if __name__ == "__main__":
    main()
