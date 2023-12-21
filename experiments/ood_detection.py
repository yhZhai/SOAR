from functools import partial
from typing import List
import argparse
import os
import os.path as osp
from pathlib import Path
import tqdm
import time

from rich import print
from matplotlib import pyplot as plt
import numpy as np
from scipy.special import xlogy
import torch
import torch.nn.functional as F
import torch.multiprocessing
from mmcv import Config, DictAction
from mmcv.runner import load_checkpoint
from mmcv.runner.fp16_utils import wrap_fp16_model
from mmcv.parallel import MMDataParallel
from mmaction.utils import register_module_hooks
from mmaction.models import build_model
from mmaction.datasets import build_dataloader, build_dataset

from experiments.utils import (
    AverageMeter,
    get_recon_error_uncertainty,
    get_evidential_learning_uncertainty,
    turn_off_pretrained,
    apply_dropout,
    update_seed,
    preprocess,
)


def parse_args():
    parser = argparse.ArgumentParser(description="MMAction2 OOD detection")
    # model config
    parser.add_argument("config", help="test config file path")
    parser.add_argument("checkpoint", help="checkpoint file")
    parser.add_argument(
        "-u",
        "--uncertainty",
        type=str,
        default="evidence",
        choices=["recon", "evidence", "bald", "entropy"],
    )
    parser.add_argument(
        "--proc", type=str, default="none", choices=["none", "shuffle", "freeze"]
    )
    parser.add_argument(
        "--evidence_type", type=str, default="exp", choices=["relu", "exp", "softplus"]
    )
    # data config
    parser.add_argument(
        "--train_data",
        help="the split file of training data",
        default="data/ucf101/ucf101_train_split_1_videos.txt",
    )
    parser.add_argument(
        "--ind_data",
        help="the split file of in-distribution testing data",
        default="data/ucf101/ucf101_val_split_1_videos.txt",
    )
    parser.add_argument(
        "--ood_data",
        help="the split file of out-of-distribution testing data",
        default="data/hmdb51/hmdb51_val_split_1_videos.txt",
    )
    # env config
    parser.add_argument("-o", "--output_name", help="output file name", default="ood")
    parser.add_argument(
        "--dense", help="dense sample frame", action="store_true", default=False
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
    parser.add_argument("--num_pass", type=int, default=10)
    opt = parser.parse_args()
    return opt


def get_results(model, data_loader, uncertainty_fn, proc: str):
    all_confidences, all_uncertainties, all_results, all_gts = [], [], [], []
    video_names = []
    model.eval()
    uncer = AverageMeter()
    pbar = tqdm.tqdm(enumerate(data_loader), total=len(data_loader))
    for _, data in pbar:
        with torch.no_grad():
            # preprocess
            data = preprocess(data, "imgs", proc)

            outputs = model(return_loss=False, **data, return_dict=True)
            score, uncertainty = uncertainty_fn(outputs=outputs)

        uncer.update(uncertainty.mean().item(), uncertainty.shape[0])
        all_uncertainties.append(uncertainty)
        preds = np.argmax(score, axis=1)
        all_results.append(preds)
        conf = np.max(score, axis=1)
        all_confidences.append(conf)
        labels = data["label"].numpy()
        all_gts.append(labels)
        video_names.extend(
            [x["filename"].split("/")[-1] for x in data["img_metas"].data[0]]
        )

        pbar.update()
        pbar.set_description(
            f"{data_loader.dataset.ann_file.split('/')[-1].split('_')[0]} "
            f"input size {list(data['imgs'].shape)} "
            f"uncertainty {uncer}"
        )
    all_confidences = np.concatenate(all_confidences, axis=0)
    all_uncertainties = np.concatenate(all_uncertainties, axis=0)
    all_results = np.concatenate(all_results, axis=0)
    all_gts = np.concatenate(all_gts, axis=0)
    video_names = np.array(video_names)

    return (
        all_confidences,
        all_uncertainties,
        all_results,
        all_gts,
        video_names,
    )


def get_stochastic_uncertainty_fn(outputs: List, method: str):
    """compute entropy or bald uncertainty

    Args:
        outputs (List): list of output
        method (str): entropy or bald
    """
    # B, C, num_pass
    outputs = torch.stack(outputs, dim=2).detach().cpu().numpy()
    # mean of all num_pass forward passes
    expect_p = np.mean(outputs, axis=2)
    # entropy of expect_p (across classes)
    entropy_expected_p = -np.sum(xlogy(expect_p, expect_p), axis=1)
    if method == "entropy":
        uncertain_score = entropy_expected_p
    elif method == "bald":
        # mean of entropies (across classes), (scalar)
        expected_entropy = -np.mean(np.sum(xlogy(outputs, outputs), axis=1), axis=-1)
        uncertain_score = entropy_expected_p - expected_entropy
    else:
        raise NotImplementedError(f"unsupported uncertainty method {method}")
    if not np.all(np.isfinite(uncertain_score)):
        uncertain_score[~np.isfinite] = 9999
    return expect_p, uncertain_score


def get_stochastic_results(
    model,
    data_loader,
    uncertainty_method: str,
    proc: str,
    num_pass: int = 10,
    apply_dp: bool = False,
    tgt_key="cls_score",
):
    all_confidences, all_uncertainties, all_results, all_gts = [], [], [], []
    video_names = []
    model.eval()
    if apply_dp:
        model.apply(apply_dropout)
    uncer = AverageMeter()
    pbar = tqdm.tqdm(enumerate(data_loader), total=len(data_loader))
    for _, data in pbar:
        all_outputs = []
        with torch.no_grad():
            # preprocess
            data = preprocess(data, "imgs", proc)

            for n in range(num_pass):
                update_seed(n)
                outputs = model(return_loss=False, **data, return_dict=True)
                all_outputs.append(outputs[tgt_key])

            score, uncertainty = get_stochastic_uncertainty_fn(
                outputs=all_outputs, method=uncertainty_method
            )

        uncer.update(uncertainty.mean().item(), uncertainty.shape[0])
        all_uncertainties.append(uncertainty)
        preds = np.argmax(score, axis=1)
        all_results.append(preds)
        conf = np.max(score, axis=1)
        all_confidences.append(conf)
        labels = data["label"].numpy()
        all_gts.append(labels)
        video_names.extend(
            [x["filename"].split("/")[-1] for x in data["img_metas"].data[0]]
        )

        pbar.update()
        pbar.set_description(
            f"{data_loader.dataset.ann_file.split('/')[-1].split('_')[0]} "
            f"input size {list(data['imgs'].shape)} "
            f"uncertainty {uncer}"
        )
    all_confidences = np.concatenate(all_confidences, axis=0)
    all_uncertainties = np.concatenate(all_uncertainties, axis=0)
    all_results = np.concatenate(all_results, axis=0)
    all_gts = np.concatenate(all_gts, axis=0)
    video_names = np.array(video_names)

    return (
        all_confidences,
        all_uncertainties,
        all_results,
        all_gts,
        video_names,
    )


def run_inference(model, datalist_file, cfg, opt):
    # change datalist and root
    cfg.data.test.ann_file = datalist_file
    cfg.data.test.data_prefix = osp.join(osp.dirname(datalist_file), "videos")

    # build the dataloader
    dataset = build_dataset(cfg.data.test, dict(test_mode=True))
    dataloader_setting = dict(
        videos_per_gpu=cfg.data.get("videos_per_gpu", 1),
        workers_per_gpu=cfg.data.get("workers_per_gpu", 1),
        dist=False,
        shuffle=False,
        pin_memory=False,  # this is critical, otherwise causes soft lockup
    )
    dataloader_setting = dict(dataloader_setting, **cfg.data.get("test_dataloader", {}))
    data_loader = build_dataloader(dataset, **dataloader_setting)

    if opt.uncertainty == "recon":
        if cfg.model.cls_head.loss_recon.type.lower() == "mseloss":
            recon_loss_fn = F.mse_loss
        elif cfg.model.cls_head.loss_recon.type.lower() == "l1loss":
            recon_loss_fn = F.l1_loss
        else:
            raise NotImplementedError
        uncertainty_fn = partial(
            get_recon_error_uncertainty, recon_loss_fn=recon_loss_fn
        )
    elif opt.uncertainty == "evidence":
        if opt.evidence_type == "relu":
            from mmaction.models.losses.edl_loss import relu_evidence as get_evidence
        elif opt.evidence_type == "exp":
            from mmaction.models.losses.edl_loss import exp_evidence as get_evidence
        elif opt.evidence_type == "softplus":
            from mmaction.models.losses.edl_loss import (
                softplus_evidence as get_evidence,
            )
        else:
            raise NotImplementedError
        uncertainty_fn = partial(
            get_evidential_learning_uncertainty,
            get_evidence_fn=get_evidence,
            num_classes=model.module.cls_head.num_classes,
        )
    elif opt.uncertainty in ["bald", "entropy"]:
        pass
    else:
        raise NotImplementedError(
            f"unsupported uncertainty estimation {opt.uncertainty}"
        )

    if opt.uncertainty in ["evidence", "recon"]:
        (
            all_confidences,
            all_uncertainties,
            all_results,
            all_gts,
            video_names,
        ) = get_results(model, data_loader, uncertainty_fn, opt.proc)
    elif opt.uncertainty in ["bald", "entropy"]:
        (
            all_confidences,
            all_uncertainties,
            all_results,
            all_gts,
            video_names,
        ) = get_stochastic_results(
            model,
            data_loader,
            opt.uncertainty,
            opt.proc,
            num_pass=opt.num_pass,
            apply_dp="dnn" in opt.config,
        )
    else:
        raise NotImplementedError(
            f"unsupported uncertainty estimation {opt.uncertainty}"
        )

    return (
        all_confidences,
        all_uncertainties,
        all_results,
        all_gts,
        video_names,
    )


def main(opt):
    # a trick to avoid ''RuntimeError: Too many open files.''
    # https://github.com/pytorch/pytorch/issues/11201
    torch.multiprocessing.set_sharing_strategy("file_system")

    # set cudnn benchmark
    torch.backends.cudnn.benchmark = True

    checkpoint_path = Path(opt.checkpoint)
    opt.output_name += f"_{checkpoint_path.stem}"
    if opt.dense:
        opt.output_name += "_dense"
    if opt.proc.lower() != "none":
        opt.output_name += f"_{opt.proc.lower()}"
        print(f"[red]using {opt.proc} preprocessing[/red]")
    opt.result_prefix = (checkpoint_path.parent / opt.output_name).as_posix()

    cfg = Config.fromfile(opt.config)

    cfg.merge_from_dict(opt.cfg_options)

    assert cfg.dataset_type == "VideoDataset", "only support video dataset"
    if opt.uncertainty in ["evidence"]:
        assert cfg.model.test_cfg.average_clips == "score"
    elif opt.uncertainty in ["entropy", "bald"]:
        assert cfg.model.test_cfg.average_clips == "prob"
    else:
        raise NotImplementedError(
            f"unsupported uncertainty estimation {opt.uncertainty}"
        )

    # run inference
    if "hmdb" in opt.ood_data.lower():
        dataset_name = "hmdb"
    elif "mit" in opt.ood_data.lower():
        dataset_name = "mit"
    elif "ucf" in opt.ood_data.lower():
        dataset_name = "ucf"
    else:
        raise NotImplementedError

    if "clean" in opt.ood_data.lower():
        dataset_name += "_clean"
    result_suffix = "_" + opt.uncertainty + "_" + dataset_name
    result_file = opt.result_prefix + result_suffix + "_result.npz"
    if not osp.exists(result_file):
        print(f"result {result_file} NOT found, running experiments")

        # The flag is used to register module's hooks
        cfg.setdefault("module_hooks", [])

        # remove redundant pretrain steps for testing
        turn_off_pretrained(cfg.model)

        # build the recognizer from a config file and checkpoint file/url
        model = build_model(cfg.model)

        if len(cfg.module_hooks) > 0:
            register_module_hooks(model, cfg.module_hooks)

        fp16_cfg = cfg.get("fp16", None)
        if fp16_cfg is not None:
            wrap_fp16_model(model)

        if osp.exists(opt.checkpoint):
            try:
                load_checkpoint(model, opt.checkpoint, map_location="cpu", strict=True)
            except Exception as e:
                assert "fc_cls" not in str(e), "fc_cls failed to load"
                print(
                    f"[red]failed to strictly load the checkpoint, setting "
                    "strict to False to re-load the checkpoint[/red]"
                )
                load_checkpoint(model, opt.checkpoint, map_location="cpu", strict=False)
            time.sleep(5)
        else:
            horizontal_line = "-" * os.get_terminal_size().columns
            print(f"[red]{horizontal_line}[/red]")
            print(f"[red]checkpoint file {opt.checkpoint} not found[/red]")
            print(
                "[red][Hint] Running random reconstruction. The checkpoint "
                "must contain a valid parent path.[/red]"
            )
            print(f"[red]{horizontal_line}[/red]")
            time.sleep(5)

        model = MMDataParallel(model, device_ids=[0])

        # prepare result path
        result_dir = osp.dirname(result_file)
        os.makedirs(result_dir, exist_ok=True)

        if not opt.dense:
            print("Using [red]SampleFrames[/red] as the sampling strategy")
            cfg.data.test.pipeline[1] = dict(
                type="SampleFrames",
                clip_len=32,
                frame_interval=2,
                num_clips=1,
                test_mode=True,
            )
        else:
            print("Using [red]DenseSampleFrames[/red] as the sampling strategy")
            cfg.data.test.pipeline[1] = dict(
                type="DenseSampleFrames",
                clip_len=32,
                frame_interval=2,
                num_clips=1,  # increasing this leads to better performance
                test_mode=True,
            )
            cfg.data.videos_per_gpu = 3
            cfg.data.workers_per_gpu = 3

        # run inference (train set)
        (
            train_confidences,
            train_uncertainties,
            train_results,
            train_labels,
            train_video_names,
        ) = run_inference(model, opt.train_data, cfg, opt)
        # run inference (OOD)
        (
            ood_confidences,
            ood_uncertainties,
            ood_results,
            ood_labels,
            ood_video_names,
        ) = run_inference(model, opt.ood_data, cfg, opt)
        # run inference (IND)
        (
            ind_confidences,
            ind_uncertainties,
            ind_results,
            ind_labels,
            ind_video_names,
        ) = run_inference(model, opt.ind_data, cfg, opt)
        if len(ind_uncertainties.shape) == 2:
            ind_uncertainties = ind_uncertainties.squeeze(1)
            ood_uncertainties = ood_uncertainties.squeeze(1)
            train_uncertainties = train_uncertainties.squeeze(1)
        # get threshold
        sorted_train_uncertainty = np.sort(train_uncertainties)[::-1]
        N = sorted_train_uncertainty.shape[0]
        topk = int(N * 0.05)
        threshold = sorted_train_uncertainty[topk]
        # save
        np.savez(
            result_file[:-4],
            train_unctt=train_uncertainties,
            ind_conf=ind_confidences,
            ood_conf=ood_confidences,
            ind_unctt=ind_uncertainties,
            ood_unctt=ood_uncertainties,
            ind_pred=ind_results,
            ood_pred=ood_results,
            ind_label=ind_labels,
            ood_label=ood_labels,
            train_video_names=train_video_names,
            ood_video_names=ood_video_names,
            ind_video_names=ind_video_names,
            threshold=threshold,
        )
    else:
        print(f"result {result_file} found, running experiments")

        results = np.load(result_file, allow_pickle=True)
        train_uncertainties = results["train_unctt"]
        ind_confidences = results["ind_conf"]
        ood_confidences = results["ood_conf"]
        ind_uncertainties = results["ind_unctt"]  # (N1,)
        ood_uncertainties = results["ood_unctt"]  # (N2,)
        ind_results = results["ind_pred"]  # (N1,)
        ood_results = results["ood_pred"]  # (N2,)
        ind_labels = results["ind_label"]
        ood_labels = results["ood_label"]
        threshold = results["threshold"]

        if len(ind_uncertainties.shape) == 2:
            ind_uncertainties = ind_uncertainties.squeeze(1)
            ood_uncertainties = ood_uncertainties.squeeze(1)
            train_uncertainties = train_uncertainties.squeeze(1)
        # get threshold
        # sorted_train_uncertainty = np.sort(train_uncertainties)[::-1]
        # N = sorted_train_uncertainty.shape[0]
        # topk = int(N * 0.05)
        # threshold = sorted_train_uncertainty[topk]

    # visualize
    ind_uncertainties = np.array(ind_uncertainties)
    ood_uncertainties = np.array(ood_uncertainties)
    train_uncertainties = np.array(train_uncertainties)
    dataName_ind = opt.ind_data.split("/")[-2].upper()
    dataName_ood = opt.ood_data.split("/")[-2].upper()
    if dataName_ind == "UCF101":
        dataName_ind = "UCF-101"
    if dataName_ood == "MIT":
        dataName_ood = "MiT-v2"
    if dataName_ood == "HMDB51":
        dataName_ood = "HMDB-51"
    plt.figure(figsize=(5, 4))  # (w, h)
    plt.rcParams["font.family"] = "Arial"  # Times New Roman
    plt.hist(
        [train_uncertainties, ind_uncertainties, ood_uncertainties],
        50,
        density=True,
        histtype="bar",
        color=["green", "blue", "red"],
        label=[
            "training set (%s)" % (dataName_ind),
            "in-distribution (%s)" % (dataName_ind),
            "out-of-distribution (%s)" % (dataName_ood),
        ],
    )
    if opt.uncertainty == "recon":
        plt.xlabel("reconstruction error")
    else:
        plt.xlabel("evidential uncertainty")
    plt.ylabel("density")
    plt.axvline(x=threshold, color="black", label="threshold", linestyle="--")
    plt.legend()
    plt.tight_layout()
    plt.savefig(opt.result_prefix + result_suffix + "_distribution.png")
    plt.close()

    print(f"result file saved to: {result_file}")


if __name__ == "__main__":

    # import cProfile
    # import pstats
    # profiler = cProfile.Profile()
    # profiler.enable()

    opt = parse_args()
    print("Running [red]{}[/red] uncertainty estimation".format(opt.uncertainty))
    main(opt)

    # profiler.disable()
    # stats = pstats.Stats(profiler).sort_stats('cumtime')
    # stats.strip_dirs()
    # stats_name = 'cprofile-data'
    # stats_name = os.path.join('tmp', stats_name)
    # stats.dump_stats(stats_name)
