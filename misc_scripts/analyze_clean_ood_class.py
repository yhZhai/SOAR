import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt


def main(datalists: str = ["data/hmdb51/hmdb_val_open_split1_video.txt",
                           "data/hmdb51/hmdb_val_open_split2_video.txt",
                           "data/hmdb51/hmdb_val_open_split3_video.txt",
                           "data/hmdb51/hmdb_val_open_split4_video.txt"],
        num_class: int = 51,
        width: float = 0.2):

    class_names = get_hmdb51_class_mapper()

    stat_list = []
    for i, datalist in enumerate(datalists):
        stat = [0] * num_class
        with open(datalist, "r") as f:
            for line in f:
                class_idx = int(line.strip().split(" ")[1])
                stat[class_idx] += 1
        stat_list.append(stat)

        plt.bar(np.arange(num_class) - width * (2 - i), stat, width=width, label=f"{i}")

    plt.xticks(np.arange(num_class), class_names.values(), rotation="vertical")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"tmp/{i}.png", dpi=300)
    plt.close()


def get_hmdb51_class_mapper(annotation_path: str = "data/hmdb51/annotations"):
    annotation_path = Path(annotation_path)
    class_names = set()
    for file in annotation_path.glob("*"):
        file_name = file.stem
        file_name = "_".join(file_name.split("_")[:-2])
        if file_name:
            class_names.add(file_name)
    
    # print(len(class_names))
    class_names = sorted(class_names)
    # print(class_names)
    return {i: class_names[i] for i in range(len(class_names))}


if __name__ == "__main__":
    main()
