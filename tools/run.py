from typing import List
import os
import subprocess
import argparse
import time
from pathlib import Path
import glob
import random
from datetime import datetime
import re

from tqdm import tqdm
from rich import print
from mmcv import DictAction, Config


wandb_config_template = """
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ],
    _delete_=True
)
"""


def main(
    opt,
    exclude_list: List[str] = ["work_dir", "resume_from", "load_from"],
    wandb_config_template=wandb_config_template,
    sleep_time: int = 5,
    debug: bool = False,
):
    if debug:
        print("[red]debug mode[/red]")
        time.sleep(1)

    # load cfg, gpu and run_name
    cfg = Config.fromfile(opt.config)
    num_gpu = len(opt.gpus.split(","))
    dist = num_gpu != 1
    if not opt.run_name.endswith("_"):
        opt.run_name = opt.run_name + "_"

    # generata arguments from cfg-options
    cfg_cmd = []
    for k, v in opt.cfg_options.items():
        cfg_cmd.append(f"{k}={v}")
    cfg_cmd = " ".join(cfg_cmd)

    # lr adjust arguments
    if not opt.no_lr_adjust:
        batch_size = cfg["data"]["videos_per_gpu"] * num_gpu
        lr = batch_size * opt.unit_lr
        print(
            f"[blue]adjusting lr from {cfg['optimizer']['lr']} to {lr} for "
            f"batch size {batch_size}[/blue]"
        )
        time.sleep(1)
        lr_adjust_cmd = f"optimizer.lr={lr}"
    else:
        lr_adjust_cmd = ""

    # resume arguments
    resume_cmd = ""
    if opt.resume:
        assert Path(opt.resume).exists()
        resume_cmd = f"--resume-from {opt.resume}"

    # deterministic arguments
    if not opt.no_deterministic:
        deterministic_cmd = f"--deterministic --seed {opt.seed}"
    else:
        random.seed(datetime.now())
        seed = random.randint(1, 2**32 - 1)
        deterministic_cmd = f"--seed {seed}"

    # automatically rename work_dir according to cfg-options
    time_stamp = time.strftime("%b-%d-%H-%M-%S", time.localtime())
    log_cmd = cfg_cmd.split(" ")
    log_cmd = list(filter(lambda x: x.split("=")[0] not in exclude_list, log_cmd))
    log_cmd = " ".join(log_cmd)
    # remove "model." to shorten dir names
    log_cmd = log_cmd.replace("model.", "")
    # remove "cls_head." to shorten dir names
    log_cmd = log_cmd.replace("cls_head.", "")
    log_cmd = log_cmd.replace("_", "").replace(" ", "_")
    work_dir = None
    if "work_dir" not in opt.cfg_options:
        work_dir = Path(cfg["work_dir"])
        work_dir = "_".join([work_dir.as_posix(), log_cmd, opt.run_name, time_stamp])
        work_dir = re.sub("_+", "_", work_dir)
    else:
        work_dir = opt.cfg_options["work_dir"]
    print(f"work_dir is set to {work_dir}")
    time.sleep(1)

    # generate a temporary config file, and overwrite wandb config
    wandb_name = opt.run_name + log_cmd + "_" + time_stamp
    wandb_name = re.sub("_+", "_", wandb_name)
    tmp_config_path = Path(opt.config)
    tmp_config_path_parent = tmp_config_path.parent / "tmp"
    tmp_config_path_parent.mkdir(exist_ok=True, parents=True)
    tmp_config_path = tmp_config_path_parent / tmp_config_path.name
    tmp_config_path = tmp_config_path.as_posix().split(".")
    tmp_config_path[0] = tmp_config_path[0] + "_" + time_stamp
    tmp_config_path = ".".join(tmp_config_path)
    with open(opt.config, "r") as f:
        config = f.readlines()
    # this is a hack, TODO replace with re
    config[0] = config[0].replace("./", "../")
    config.append(wandb_config_template.format(name=wandb_name))
    with open(tmp_config_path, "w") as f:
        f.writelines(config)
    print(f"creating temporary config file {tmp_config_path}")

    if dist:  # distributed training
        cmd = (
            f"NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES={opt.gpus} "
            f"PORT={opt.port} zsh "
            f"tools/dist_train.sh {tmp_config_path} {num_gpu} "
            f"--validate {deterministic_cmd} {resume_cmd}"
            f"--cfg-options work_dir={work_dir} {cfg_cmd} {lr_adjust_cmd}"
        )
    else:  # non-distributed training
        cmd = (
            f"CUDA_VISIBLE_DEVICES={opt.gpus} python tools/train.py "
            f"{tmp_config_path} --validate {deterministic_cmd} "
            f"--gpu-ids 0 {resume_cmd} --cfg-options "
            f"work_dir={work_dir} {cfg_cmd} {lr_adjust_cmd}"
        )

    horizontal_line = "+" * os.get_terminal_size().columns
    print(f"executing command:\n{horizontal_line}\n{cmd}\n{horizontal_line}")
    pbar = tqdm(reversed(range(sleep_time)), total=sleep_time)
    for t in pbar:
        pbar.set_description(f"running in {t + 1} seconds")
        time.sleep(1)
    run_command(cmd, debug)

    # open-set evaluation
    if not opt.no_eval:
        # OOD detection
        print(horizontal_line)
        gpu_id = opt.gpus.replace(",", " ")[0]
        checkpoint_path = (Path(work_dir) / "latest.pth").as_posix()
        config_path = glob.glob(work_dir + "*.py")
        if len(config_path) > 0:
            config_path = config_path[0]
        else:
            config_path = tmp_config_path
        hmdb_ood_detection_cmd = (
            f"CUDA_VISIBLE_DEVICES={gpu_id} python experiments/ood_detection.py "
            f"{opt.eval_config_path} {checkpoint_path} --ood_data "
            "data/hmdb51/hmdb51_val_split_1_videos.txt "
            "-u evidence --dense"  # ignore reconstruction
        )
        run_command(hmdb_ood_detection_cmd, debug)
        mit_ood_detection_cmd = (
            f"CUDA_VISIBLE_DEVICES={gpu_id} python experiments/ood_detection.py "
            f"{opt.eval_config_path} {checkpoint_path} --ood_data "
            "data/mit/mit_val_list_videos.txt "
            "-u evidence --dense"  # ignore reconstruction
        )
        run_command(mit_ood_detection_cmd, debug)

        # run openness
        print(horizontal_line)
        glob_tgt = Path(work_dir) / "*.npz"
        print(f"Searching for {glob_tgt.as_posix()}")
        result_files = glob.glob(glob_tgt.as_posix())
        print(f"Running open-set evaluation on {result_files}")
        for result_file in result_files:
            if "hmdb" in result_file.lower():
                run_openness_cmd = (
                    f"python experiments/open_set_evaluation.py {result_file} --clean"
                )
                run_command(run_openness_cmd, debug)

            run_openness_cmd = (
                f"python experiments/open_set_evaluation.py {result_file}"
            )
            run_command(run_openness_cmd, debug)

    os.remove(tmp_config_path)


def run_command(command: str, debug: bool):
    if debug:
        return
    subprocess.run(command, shell=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Running wrapper")
    parser.add_argument("config", help="train config file path")
    parser.add_argument("--gpus", type=str, help="gpu ids split by comma (,)")
    parser.add_argument("--port", type=int, default=29500)
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--no_eval", action="store_true", default=False)
    parser.add_argument("--no_lr_adjust", action="store_true", default=False)
    parser.add_argument("-nd", "--no_deterministic", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--unit_lr", type=float, default=0.001 / 8)
    parser.add_argument(
        "-ecp",
        "--eval_config_path",
        type=str,
        default="configs/recognition/i3d/inference_i3d_enn.py",
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        default={},
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file. For example, "
        "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'",
    )
    opt = parser.parse_args()

    main(opt, debug=False)
