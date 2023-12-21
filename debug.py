from collections import OrderedDict
from pathlib import Path
import shutil
import subprocess

from tqdm import tqdm
import math
import numpy as np
import torch
# from spatial_correlation_sampler import (
#     SpatialCorrelationSampler,
#     spatial_correlation_sample,
# )
from matplotlib import pyplot as plt


def compare_npz(
    original_path="work_dirs/finetune_ucf101_i3d_edlnokl_avuc_debias-trained-by-myself/obsolete/ood_result.npz",
    ours_path="work_dirs/finetune_ucf101_i3d_edlnokl_avuc_debias-trained-by-myself/ood_result.npz",
    original_trainset_uncertainty_path="work_dirs/finetune_ucf101_i3d_edlnokl_avuc_debias-trained-by-myself/I3D_EDLNoKL_EDL_trainset_uncertainties.npz",
):
    original = np.load(original_path, allow_pickle=True)
    ours = np.load(ours_path, allow_pickle=True)
    original_trainset_uncertainty = np.load(
        original_trainset_uncertainty_path, allow_pickle=True
    )
    original_ind_uncertainty = original["ind_unctt"]
    ours_ind_uncertainty = ours["ind_unctt"]
    original_trainset_uncertainty = original_trainset_uncertainty["uncertainty"]

    sorted_ours_ind_uncertainty = np.sort(ours_ind_uncertainty)[::-1]
    N = sorted_ours_ind_uncertainty.shape[0]
    tgt = 0.004551
    for i in range(N - 1):
        if (
            sorted_ours_ind_uncertainty[i] >= tgt
            and sorted_ours_ind_uncertainty[i + 1] <= tgt
        ):
            print(sorted_ours_ind_uncertainty[i], i, N, i / N)
            break


def correlation():
    device = "cuda"
    b, c, t, h, w = 4, 3, 32, 64, 64
    dtype = torch.float32

    input1 = torch.randn((b, c, h, w), dtype=dtype, device=device, requires_grad=True)
    input2 = torch.randn_like(input1)

    out = spatial_correlation_sample(
        input1,
        input2,
        kernel_size=3,
        patch_size=21,
        stride=1,
        padding=1,
        dilation=1,
        dilation_patch=2,
    )

    print(input1.shape, input2.shape, out.shape)


def create_new_checkpoint(
    original="pretrained/i3d_r50_dense_256p_32x2x1_100e_kinetics400_rgb_20200725-24eb54cc.pth",
    pretrain="work_dirs/i3d_r50_32x2x1_100e_ucf101_rgb_edl_dis_heavy_pretrain_dis_heavy_pretrain_Aug-02-18-58/epoch_50.pth",
    reference="work_dirs/i3d_r50_32x2x1_100e_ucf101_rgb_edl_disentangle_heavy_model.clshead.lossdebias.lossweight=0.001_edl_dis_heavy_Aug-02-00-43/epoch_50.pth",
    output_path="pretrained/edl_dis_heavy_pretrain.pth",
):
    original = torch.load(original, map_location="cpu")
    pretrain = torch.load(pretrain, map_location="cpu")
    reference = torch.load(reference, map_location="cpu")

    original_state_dict = original["state_dict"]
    pretrain_state_dict = pretrain["state_dict"]
    reference_state_dict = reference["state_dict"]
    output_state_dict = OrderedDict()
    for k in reference_state_dict.keys():
        if k.startswith("backbone"):
            output_state_dict[k] = pretrain_state_dict[k]
        elif k.startswith("cls_head.cls_backbone"):
            output_state_dict[k] = original_state_dict[k[13:]]
        else:
            print(f"skipping key {k}")

    pretrain["state_dict"] = output_state_dict
    torch.save(pretrain, output_path)


def cost_volume():
    h = int(112 / 2)
    w = int(112 / 2)
    off_template_w = np.zeros((h, w, w), dtype=np.float32)
    off_template_h = np.zeros((h, w, h), dtype=np.float32)
    for ii in range(h):
        for jj in range(w):
            for i in range(h):
                off_template_h[ii, jj, i] = i - ii
            for j in range(w):
                off_template_w[ii, jj, j] = j - jj
    m = np.reshape(off_template_w, newshape=(h * w, w))[None, :, :] * 2
    v = np.reshape(off_template_h, newshape=(h * w, h))[None, :, :] * 2
    print("a")


def cosine_annealing():
    peak = 1
    num_epoch = 50
    stop_epoch = 50
    v = []
    # k = torch.linspace(-0.5, 0.5, num_epoch)
    # for i in range(num_epoch):
    #     v.append(torch.sin(k[i] * math.pi).item() * peak)
    for i in range(num_epoch):
        if i <= stop_epoch:
            v.append(math.sin((i - stop_epoch / 2) / stop_epoch * math.pi))
        else:
            v.append(peak)
    print(v)
    plt.plot(v)
    plt.xlabel("epoch")
    plt.ylabel("alpha")
    # plt.plot(k * 2)
    plt.savefig("tmp/tmp2.png")


def read_config(
    config_path1: str = "tmp/config1.py",
    config_path2: str = "tmp/config2.py",
):
    eval("exec('import {} as config1')".format(config_path1.replace(".py", "").replace("/", ".")))
    eval("exec('import {} as config2')".format(config_path2.replace(".py", "").replace("/", ".")))
    print("a")


def pcc():
    from scipy.stats import pearsonr
    x = np.array([0, 1, 2, 3, 4])
    y = np.array([0, 1, 2, 3, 4])
    y = y / 10
    pcc = pearsonr(x, y)
    print(pcc)


def rename(src_path: str = "data/mit_scene_feature/validation", tgt_path: str = "data/validation"):
    tgt_path = Path(tgt_path)
    tgt_path.mkdir(parents=True, exist_ok=True)
    src_path = Path(src_path)
    for file in tqdm(src_path.rglob("*.npy"), total=30473):
        new_file = file.as_posix()[:-8] + file.suffix
        new_file = new_file.replace("mit_scene_feature/", "")
        new_file = Path(new_file)
        new_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file.as_posix(), new_file.as_posix())


def run_clean_openness_eval(tgt_path: str = "work_dirs"):
    tgt_path = Path(tgt_path)
    subdir_list = list(tgt_path.iterdir())
    for subdir in tqdm(subdir_list):
        result_file = subdir / "ood_latest_evidence_hmdb_result.npz"
        if result_file.exists():
            command = f"python experiments/open_set_evaluation.py {result_file.as_posix()} --clean" 
            subprocess.run(command, shell=True)
        else:
            print("\nno result found in {}\n".format(subdir))


def create_diving48_training_subset(input_path, output_path):
    # read the file
    with open(input_path, 'r') as file:
        lines = file.readlines()

    # filter lines with number < 24
    filtered_lines = [line for line in lines if int(line.split()[-1]) < 24]

    # write the filtered lines to the output file
    with open(output_path, 'w') as file:
        file.writelines(filtered_lines)


if __name__ == "__main__":
    create_diving48_training_subset("data/diving48/diving48_train_list_videos.txt",
                                    "data/diving48/diving48_train_list_videos_24classes.txt")
