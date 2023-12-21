from functools import partial
import argparse
from pathlib import Path

import tqdm
from rich import print
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import mmcv
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
    UnNormalize,
    minmax_normalization,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Reconstruction visualization")
    # model config
    parser.add_argument("config", help="test config file path")
    parser.add_argument("checkpoint", help="checkpoint file")
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
    parser.add_argument(
        "--output", help="output folder name", default="cams_visualization"
    )
    parser.add_argument(
        "--individual",
        action="store_true",
        default=False,
        help="save individual figures",
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


def get_results(
    model,
    data_loader,
    folder: Path,
    uncertainty_fn,
    recon_loss_fn,
    unnormalize_fn,
    individual: bool,
):
    pbar = tqdm.tqdm(enumerate(data_loader), total=len(data_loader))
    model.eval()
    for i, data in pbar:
        frame_dir = Path(data["img_metas"].data[0][0]["frame_dir"]).name

        with torch.no_grad():
            outputs = model(return_loss=False, **data, return_dict=True, get_cams=True)

            recon_error_map = recon_loss_fn(
                outputs["recon"], outputs["gt_recon"], reduction="none"
            )
            recon_error_map = unnormalize_fn(recon_error_map) / 255.0
            recon_error_map = recon_error_map.clamp(0, 1)
            frame_list = []
            recon_list = []
            recon_error_map_list = []
            scene_cam_list = []
            for t in range(outputs["gt_recon"].shape[2]):
                # frames
                frame = unnormalize_fn(outputs["gt_recon"][0, :, t, ...]) / 255.0
                frame = frame.permute(1, 2, 0).clamp(0, 1)
                frame = frame.detach().cpu().numpy()
                frame_list.append(frame)

                # recon results
                recon = unnormalize_fn(outputs["recon"][0, :, t, ...]) / 255.0
                recon = recon.permute(1, 2, 0).clamp(0, 1)
                recon = recon.detach().cpu().numpy()
                recon_list.append(recon)

                # recon error maps
                recon_error_map_list.append(
                    recon_error_map[0, :, t, ...]
                    .permute(1, 2, 0)
                    .detach()
                    .cpu()
                    .numpy()
                )

            # uncertainty cam from evidential cam
            outputs["cls_cam"] = F.interpolate(
                outputs["cls_cam"], outputs["recon"].shape[2:], mode="trilinear"
            )
            # num_crop, t, h, w
            uncertainty_cam = uncertainty_fn(outputs=outputs, tgt_key="cls_cam")[-1]
            uncertainty_cam_list = []
            for t in range(uncertainty_cam.shape[1]):
                uncertainty_cam_list.append(uncertainty_cam[0, t, ...])

            # scene cam
            if "scene_cam" in outputs.keys() and outputs["scene_cam"] is not None:
                outputs["scene_cam"] = outputs["scene_cam"].softmax(dim=1)
                outputs["scene_cam"] = F.interpolate(
                    outputs["scene_cam"], outputs["recon"].shape[2:], mode="trilinear"
                )
                scene_label = data["scene_pred"].squeeze(0).argmax()
                # t, h, w
                scene_cam = (
                    outputs["scene_cam"][0, scene_label, ...].detach().cpu().numpy()
                )
                scene_cam_list = []
                for t in range(scene_cam.shape[0]):
                    scene_cam_list.append(scene_cam[t, ...])

            # save individual files
            (folder / frame_dir).mkdir(exist_ok=True, parents=True)
            if individual:
                (folder / frame_dir / "frames").mkdir(exist_ok=True, parents=True)
                (folder / frame_dir / "recons").mkdir(exist_ok=True, parents=True)
                (folder / frame_dir / "recon_error_maps").mkdir(
                    exist_ok=True, parents=True
                )
                (folder / frame_dir / "uncertainty_maps").mkdir(
                    exist_ok=True, parents=True
                )
                (folder / frame_dir / "scene_cams").mkdir(exist_ok=True, parents=True)
                for t in range(uncertainty_cam.shape[1]):
                    plt.imshow(frame_list[t])
                    plt.axis("off")
                    plt.savefig(
                        (folder / frame_dir / "frames" / f"{t}.png").as_posix(),
                        dpi=400,
                        bbox_inches="tight",
                        transparent=True,
                        pad_inches=0.0,
                    )
                    plt.close()

                    plt.imshow(recon_list[t])
                    plt.axis("off")
                    plt.savefig(
                        (folder / frame_dir / "recons" / f"{t}.png").as_posix(),
                        dpi=400,
                        bbox_inches="tight",
                        transparent=True,
                        pad_inches=0.0,
                    )
                    plt.close()

                    plt.imshow(minmax_normalization(recon_error_map_list[t]))
                    plt.axis("off")
                    plt.savefig(
                        (
                            folder / frame_dir / "recon_error_maps" / f"{t}.png"
                        ).as_posix(),
                        dpi=400,
                        bbox_inches="tight",
                        transparent=True,
                        pad_inches=0.0,
                    )
                    plt.close()

                    plt.imshow(minmax_normalization(uncertainty_cam_list[t]))
                    plt.axis("off")
                    plt.savefig(
                        (
                            folder / frame_dir / "uncertainty_maps" / f"{t}.png"
                        ).as_posix(),
                        dpi=400,
                        bbox_inches="tight",
                        transparent=True,
                        pad_inches=0.0,
                    )
                    plt.close()

                    if (
                        "scene_cam" in outputs.keys()
                        and outputs["scene_cam"] is not None
                    ):
                        plt.imshow(minmax_normalization(scene_cam_list[t]))
                        plt.axis("off")
                        plt.savefig(
                            (folder / frame_dir / "scene_cams" / f"{t}.png").as_posix(),
                            dpi=400,
                            bbox_inches="tight",
                            transparent=True,
                            pad_inches=0.0,
                        )
                        plt.close()

            # concat to a single matrix for plotting
            frame_list = np.concatenate(frame_list, axis=1)
            recon_list = np.concatenate(recon_list, axis=1)
            recon_error_map_list = np.concatenate(recon_error_map_list, axis=1).mean(
                axis=2
            )
            recon_error_map_list = minmax_normalization(recon_error_map_list)
            uncertainty_cam_list = np.concatenate(uncertainty_cam_list, axis=1)
            uncertainty_cam_list = minmax_normalization(uncertainty_cam_list)
            if "scene_cam" in outputs.keys() and outputs["scene_cam"] is not None:
                scene_cam_list = np.concatenate(scene_cam_list, axis=1)
                scene_cam_list = minmax_normalization(scene_cam_list)

            # draw
            plt.subplot(511).set_title("raw frame")
            plt.imshow(frame_list)
            plt.axis("off")
            plt.subplot(512).set_title("recon result")
            plt.imshow(recon_list)
            plt.axis("off")
            plt.subplot(513).set_title("recon error map")
            plt.imshow(recon_error_map_list)
            plt.axis("off")
            plt.subplot(514).set_title("uncertainty map")
            im = plt.imshow(uncertainty_cam_list)
            plt.axis("off")
            plt.colorbar(im, orientation="horizontal")
            if "scene_cam" in outputs.keys() and outputs["scene_cam"] is not None:
                plt.subplot(515).set_title("scene cam")
                im = plt.imshow(scene_cam_list)
                plt.axis("off")
                plt.colorbar(im, orientation="horizontal")

            plt.subplots_adjust(wspace=0, hspace=0)
            plt.tight_layout()
            plt.savefig(
                (folder / frame_dir / "result.png").as_posix(),
                dpi=400,
                bbox_inches="tight",
            )
            plt.close()


def run_inference(model, datalist_file, cfg, opt, subfolder: Path):
    subfolder.mkdir(parents=True, exist_ok=True)

    # switch config for different dataset
    cfg.data.test.ann_file = datalist_file
    cfg.data.test.data_prefix = (Path(datalist_file).parent / "rawframes").as_posix()
    cfg.data.test.pipeline[0] = dict(
        type="SampleFrames", clip_len=32, frame_interval=2, num_clips=1, test_mode=True
    )
    cfg.data.test.pipeline[3] = dict(type="CenterCrop", crop_size=256)

    # build the dataloader
    dataset = build_dataset(cfg.data.test, dict(test_mode=True))
    dataloader_setting = dict(
        videos_per_gpu=cfg.data.get("videos_per_gpu", 1),
        workers_per_gpu=cfg.data.get("workers_per_gpu", 1),
        dist=False,
        shuffle=False,
        pin_memory=False,
    )
    dataloader_setting = dict(dataloader_setting, **cfg.data.get("test_dataloader", {}))
    data_loader = build_dataloader(dataset, **dataloader_setting)

    if cfg.model.cls_head.loss_recon.type.lower() == "mseloss" or (
        cfg.model.cls_head.loss_recon.type.lower() == "weightedloss"
        and cfg.model.cls_head.loss_recon.loss_impl == "mse"
    ):
        recon_loss_fn = F.mse_loss
    elif "l1loss" in cfg.model.cls_head.loss_recon.type.lower() or (
        cfg.model.cls_head.loss_recon.type.lower() == "weightedloss"
        and cfg.model.cls_head.loss_recon.loss_impl == "l1"
    ):
        recon_loss_fn = F.l1_loss
    else:
        raise NotImplementedError(
            f"recon loss {cfg.model.cls_head.loss_recon.type.lower()} not supported"
        )

    if cfg.model.test_cfg.evidence_type == "relu":
        from mmaction.models.losses.edl_loss import relu_evidence as get_evidence
    elif cfg.model.test_cfg.evidence_type == "exp":
        from mmaction.models.losses.edl_loss import exp_evidence as get_evidence
    elif cfg.model.test_cfg.evidence_type == "softplus":
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

    unnormalize_fn = UnNormalize(
        mean=cfg.img_norm_cfg["mean"], std=cfg.img_norm_cfg["std"]
    )
    get_results(
        model,
        data_loader,
        subfolder,
        uncertainty_fn,
        recon_loss_fn,
        unnormalize_fn,
        opt.individual,
    )


def main(opt):

    checkpoint_path = Path(opt.checkpoint)
    opt.output += f"_{checkpoint_path.stem}"
    vis_path = checkpoint_path.parent / opt.output
    vis_path.mkdir(parents=True, exist_ok=True)

    cfg = Config.fromfile(opt.config)

    cfg.merge_from_dict(opt.cfg_options)

    # set cudnn benchmark
    torch.backends.cudnn.benchmark = True
    cfg.data.test.test_mode = True

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
    load_checkpoint(model, opt.checkpoint, map_location="cpu", strict=True)
    model = MMDataParallel(model, device_ids=[0])

    # run inference (train set)
    run_inference(model, opt.train_data, cfg, opt, vis_path / "train")
    # run inference (OOD)
    run_inference(model, opt.ood_data, cfg, opt, vis_path / "ood_test")
    # run inference (IND)
    run_inference(model, opt.ind_data, cfg, opt, vis_path / "ind_test")


if __name__ == "__main__":

    opt = parse_args()
    main(opt)
