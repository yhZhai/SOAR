from functools import partial
import argparse
import os
import os.path as osp
from pathlib import Path
import tqdm
import time
import copy

import numpy as np
import torch
import torch.nn.functional as F
from rich import print
from matplotlib import pyplot as plt
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
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
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze the relationship between reconstruction loss and uncertainty."
    )
    # model config
    parser.add_argument("config", help="test config file path")
    parser.add_argument("checkpoint", help="checkpoint file")
    parser.add_argument(
        "--evidence_type", type=str, default="exp", choices=["relu", "exp", "softplus"]
    )
    parser.add_argument(
        "-dt", "--dataset_type", type=str, default="frame", choices=["video", "frame"]
    )
    # data config
    parser.add_argument(
        "--train_data",
        help="the split file of training data",
        default="data/ucf101/ucf101_train_split_1_rawframes.txt",
    )
    parser.add_argument(
        "--ind_data",
        help="the split file of in-distribution testing data",
        default="data/ucf101/ucf101_val_split_1_rawframes.txt",
    )
    parser.add_argument(
        "--ood_data",
        help="the split file of out-of-distribution testing data",
        default="data/hmdb51/hmdb51_val_split_1_rawframes.txt",
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
    opt = parser.parse_args()
    return opt


def get_results(model, data_loader, uncertainty_fn, recon_error_fn):

    all_recon_error, all_uncertainty, all_video_names = [], [], []
    pbar = tqdm.tqdm(enumerate(data_loader), total=len(data_loader))
    model.eval()
    for i, data in pbar:
        with torch.no_grad():
            outputs = model(return_loss=False, **data, return_dict=True)
            recon_error = recon_error_fn(outputs=outputs)[-1]
            uncertainty = uncertainty_fn(outputs=outputs)[-1]

        all_recon_error.append(recon_error)
        all_uncertainty.append(uncertainty)

        if "img_metas" in data.keys():
            all_video_names.append(
                data["img_metas"].data[0][0]["filename"].split("/")[-1]
            )
        else:
            all_video_names.append(str(i))

        pbar.update()
        pbar.set_description(
            f"dataset {data_loader.dataset.ann_file.split('/')[-1].split('_')[0]} "
        )
    all_uncertainty = np.concatenate(all_uncertainty, axis=0)
    all_recon_error = np.concatenate(all_recon_error, axis=0)
    all_video_names = np.array(all_video_names)

    return all_uncertainty, all_recon_error, all_video_names


def run_inference(model, datalist_file, cfg, opt):
    # switch config for different dataset
    print(
        f"using [red]{opt.dataset_type}[/red] dataset on {datalist_file.split('/')[-1]}"
    )

    # prepare the dataset
    if opt.dataset_type == "video":
        cfg.data.test.ann_file = datalist_file.replace("rawframes", "videos")
        cfg.data.test.data_prefix = osp.join(osp.dirname(datalist_file), "videos")
        for key in ["scene_root", "scene_feature_root", "scene_pred_root"]:
            if key in cfg.data.test.keys():
                cfg.data.test.pop(key)
    elif opt.dataset_type == "frame":
        cfg.data.test.ann_file = datalist_file
        cfg.data.test.data_prefix = osp.join(osp.dirname(datalist_file), "rawframes")
    else:
        raise NotImplementedError(f"Unsupported dataset type {opt.dataset_type}")

    # build the dataloader
    dataset = build_dataset(cfg.data.test, dict(test_mode=True))
    dataloader_setting = dict(
        videos_per_gpu=cfg.data.get("videos_per_gpu", 1),
        workers_per_gpu=cfg.data.get("workers_per_gpu", 2),
        dist=False,
        shuffle=False,
        pin_memory=False,  # this is critical, otherwise causes soft lockup
    )
    dataloader_setting = dict(dataloader_setting, **cfg.data.get("test_dataloader", {}))
    data_loader = build_dataloader(dataset, **dataloader_setting)

    # get functions for reconstruction error and uncertainty
    if cfg.model.cls_head.loss_recon.type.lower() == "mseloss":
        recon_loss_fn = F.mse_loss
    elif cfg.model.cls_head.loss_recon.type.lower() == "l1loss":
        recon_loss_fn = F.l1_loss
    else:
        raise NotImplementedError
    recon_error_fn = partial(get_recon_error_uncertainty, recon_loss_fn=recon_loss_fn)

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

    (all_uncertainty, all_recon_error, video_names) = get_results(
        model, data_loader, uncertainty_fn, recon_error_fn
    )
    return all_uncertainty, all_recon_error, video_names


def main(opt):
    # a trick to avoid ''RuntimeError: Too many open files.''
    # https://github.com/pytorch/pytorch/issues/11201
    import torch.multiprocessing

    torch.multiprocessing.set_sharing_strategy("file_system")

    opt = copy.deepcopy(opt)

    checkpoint_path = Path(opt.checkpoint)
    opt.output_name += f"_{checkpoint_path.stem}"
    if opt.dense:
        opt.output_name += "_dense"
    opt.result_prefix = (checkpoint_path.parent / opt.output_name).as_posix()

    cfg = Config.fromfile(opt.config)

    cfg.merge_from_dict(opt.cfg_options)

    # set cudnn benchmark
    torch.backends.cudnn.benchmark = True
    cfg.data.test.test_mode = True

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
    result_suffix = "_" + dataset_name
    result_file = opt.result_prefix + result_suffix + "scatter_result.npz"

    if not osp.exists(result_file):
        print(f"result {result_file} NOT found, running experiments")
        # The flag is used to register module's hooks
        cfg.setdefault("module_hooks", [])

        # remove redundant pretrain steps for testing
        turn_off_pretrained(cfg.model)

        # build the recognizer from a config file and checkpoint file/url
        model = build_model(cfg.model, train_cfg=None, test_cfg=cfg.get("test_cfg"))

        if len(cfg.module_hooks) > 0:
            register_module_hooks(model, cfg.module_hooks)

        fp16_cfg = cfg.get("fp16", None)
        if fp16_cfg is not None:
            wrap_fp16_model(model)

        if osp.exists(opt.checkpoint):
            load_checkpoint(model, opt.checkpoint, map_location="cpu", strict=True)
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

        # change dataset pipeline according to dataset type
        sample_pipeline_idx = 0
        if opt.dataset_type == "video":
            # assume the config file is a frame dataset
            sample_pipeline_idx = 1
            cfg.data.test.type = "VideoDataset"
            cfg.data.test.pipeline[-2] = dict(
                type="Collect", keys=["imgs", "label"], meta_keys=["filename"]
            )
            cfg.data.test.pipeline[1] = dict(type="DecordDecode")
            cfg.data.test.pipeline.insert(0, dict(type="DecordInit"))

        if not opt.dense:
            print("Using [red]SampleFrames[/red] as the sampling strategy")
            cfg.data.test.pipeline[sample_pipeline_idx] = dict(
                type="SampleFrames",
                clip_len=32,
                frame_interval=2,
                num_clips=1,
                test_mode=True,
            )
        else:
            print("Using [red]DenseSampleFrames[/red] as the sampling strategy")
            cfg.data.test.pipeline[sample_pipeline_idx] = dict(
                type="DenseSampleFrames",
                clip_len=32,
                frame_interval=2,
                num_clips=1,
                test_mode=True,
            )

        # run inference (train set)
        (train_uncertainty, train_recon_error, train_video_names) = run_inference(
            model, opt.train_data, cfg, opt
        )
        # run inference (OOD)
        (ood_uncertainty, ood_recon_error, ood_video_names) = run_inference(
            model, opt.ood_data, cfg, opt
        )
        # # run inference (IND)
        (ind_uncertainty, ind_recon_error, ind_video_names) = run_inference(
            model, opt.ind_data, cfg, opt
        )
        if len(ind_uncertainty.shape) == 2:
            ind_uncertainty = ind_uncertainty.squeeze(1)
            ood_uncertainty = ood_uncertainty.squeeze(1)
            train_uncertainty = train_uncertainty.squeeze(1)
        if len(ind_recon_error.shape) == 2:
            ind_recon_error = ind_recon_error.squeeze(1)
            ood_recon_error = ood_recon_error.squeeze(1)
            train_recon_error = train_recon_error.squeeze(1)

        np.savez(
            result_file[:-4],
            train_uncertainty=train_uncertainty,
            train_recon_error=train_recon_error,
            train_video_names=train_video_names,
            ood_uncertainty=ood_uncertainty,
            ood_recon_error=ood_recon_error,
            ood_video_names=ood_video_names,
            ind_uncertainty=ind_uncertainty,
            ind_recon_error=ind_recon_error,
            ind_video_names=ind_video_names,
        )
    else:
        print(f"result {result_file} found, running experiments")
        results = np.load(result_file, allow_pickle=True)
        train_uncertainty = results["train_uncertainty"]
        train_recon_error = results["train_recon_error"]
        train_video_names = results["train_video_names"]
        ood_uncertainty = results["ood_uncertainty"]
        ood_recon_error = results["ood_recon_error"]
        ood_video_names = results["ood_video_names"]
        ind_uncertainty = results["ind_uncertainty"]
        ind_recon_error = results["ind_recon_error"]
        ind_video_names = results["ind_video_names"]
        if len(ind_uncertainty.shape) == 2:
            ind_uncertainty = ind_uncertainty.squeeze(1)
            ood_uncertainty = ood_uncertainty.squeeze(1)
            train_uncertainty = train_uncertainty.squeeze(1)
        if len(ind_recon_error.shape) == 2:
            ind_recon_error = ind_recon_error.squeeze(1)
            ood_recon_error = ood_recon_error.squeeze(1)
            train_recon_error = train_recon_error.squeeze(1)

    # visualize
    dataName_ind = opt.ind_data.split("/")[-2].upper()
    dataName_ood = opt.ood_data.split("/")[-2].upper()
    if dataName_ind == "UCF101":
        dataName_ind = "UCF-101"
    if dataName_ood == "MIT":
        dataName_ood = "MiT-v2"
    if dataName_ood == "HMDB51":
        dataName_ood = "HMDB-51"

    plt.scatter(ood_uncertainty, ood_recon_error, label="ood")
    plt.plot(
        np.unique(ood_uncertainty),
        np.poly1d(np.polyfit(ood_uncertainty, ood_recon_error, 1))(
            np.unique(ood_uncertainty)
        ),
    )
    plt.legend()
    plt.xlabel("uncertainty")
    plt.ylabel("recon error")
    plt.tight_layout()
    plt.savefig(opt.result_prefix + result_suffix + "_ood_scatter.png", dpi=300)
    plt.close()

    plt.scatter(ind_uncertainty, ind_recon_error, label="ind")
    plt.plot(
        np.unique(ind_uncertainty),
        np.poly1d(np.polyfit(ind_uncertainty, ind_recon_error, 1))(
            np.unique(ind_uncertainty)
        ),
    )
    plt.legend()
    plt.xlabel("uncertainty")
    plt.ylabel("recon error")
    plt.tight_layout()
    plt.savefig(opt.result_prefix + result_suffix + "_ind_scatter.png", dpi=300)
    plt.close()

    plt.scatter(train_uncertainty, train_recon_error, label="train")
    plt.plot(
        np.unique(train_uncertainty),
        np.poly1d(np.polyfit(train_uncertainty, train_recon_error, 1))(
            np.unique(train_uncertainty)
        ),
    )
    plt.legend()
    plt.xlabel("uncertainty")
    plt.ylabel("recon error")
    plt.tight_layout()
    plt.savefig(opt.result_prefix + result_suffix + "_train_scatter.png", dpi=300)
    plt.close()

    plt.scatter(train_uncertainty, train_recon_error, label="train")
    plt.scatter(ood_uncertainty, ood_recon_error, label="ood")
    plt.scatter(ind_uncertainty, ind_recon_error, label="ind")
    plt.legend()
    plt.xlabel("uncertainty")
    plt.ylabel("recon error")
    plt.tight_layout()
    plt.savefig(opt.result_prefix + result_suffix + "_all_scatter.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    opt = parse_args()
    main(opt)
