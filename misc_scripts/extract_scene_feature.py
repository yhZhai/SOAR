"""
Scene feature extraction usage:
python misc_scripts/extract_scene_feature.py {config_path} --out
    {output_folder_path} --extract_tgt {feat or label}

[Hint #1] The config_path is typically
    configs/recognition/resnet50_scene/resnet50_ucf101_rgb_scene_feature.py.
[Hint #2] Remember to extract feature for both training and testing sets by
    setting ann_file_val.
[Hint #3] The model used to extract feature is located at
    pretrained/resnet50_places365.pth.
"""
import argparse
from pathlib import Path

from tqdm import tqdm
from rich import print
import numpy as np
import torch
import torch.nn.functional as F
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel
from mmcv.runner.fp16_utils import wrap_fp16_model
from mmaction.datasets import build_dataloader, build_dataset
from mmaction.models import build_model

from experiments.utils import AverageMeter


def parse_args():
    parser = argparse.ArgumentParser(
        description="MMAction2 clip-level feature extraction"
    )
    parser.add_argument("config", help="test config file path")
    parser.add_argument("--out", help="output folder name")
    parser.add_argument(
        "--extract_tgt",
        type=str,
        default="feat",
        choices=["feat", "label"],
        help="extract final feature or label",
    )
    parser.add_argument(
        "--pretrained_path", type=str, default="pretrained/resnet50_places365.pth"
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


def single_gpu_test(model, data_loader, out_parent: str):  # noqa: F811
    """Test model with a single gpu.

    This method tests model with a single gpu and
    displays test progress bar.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.

    Returns:
        list: The prediction results.
    """
    out_parent = Path(out_parent)
    out_parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    max_act, min_act = AverageMeter(), AverageMeter()
    pbar = tqdm(total=len(data_loader))
    for i, data in enumerate(data_loader):
        filename = data.pop("img_metas")
        filename = filename.data[0][0]["filename"]
        filename = Path(filename)

        data["imgs"] = data["imgs"].squeeze(3)
        with torch.no_grad():
            # numpy results averaged in the TSNHead
            result = model(return_loss=False, **data)

        if model.module.feature_extraction:
            desc_suffix = ""
        else:
            softmax_output = F.softmax(torch.tensor(result), dim=1)
            max_act.update(softmax_output.max().item())
            min_act.update(softmax_output.min().item())
            desc_suffix = (
                f" max: {softmax_output.max().item():.7f} (ema: "
                f"{max_act}), ema min: "
                f"{softmax_output.min().item():.7f} (ema: {min_act})"
            )

        filename = out_parent / filename.relative_to(filename.parents[1]).with_suffix(
            ".npy"
        )
        filename.parent.mkdir(parents=True, exist_ok=True)
        np.save(filename.as_posix(), result)
        pbar.set_description(f"output size {result.shape}{desc_suffix}")
        pbar.update()


def inference_pytorch(opt, cfg, data_loader):
    """Get predictions by pytorch models."""
    if opt.extract_tgt == "feat":
        test_cfg = dict(average_clips=None, feature_extraction=True, _delete_=True)
    elif opt.extract_tgt == "label":
        test_cfg = dict(average_clips=None, feature_extraction=False, _delete_=True)
    else:
        raise NotImplementedError(f"extraction target {opt.extract_tgt} not supported")
    cfg.model.test_cfg = test_cfg
    # build the model and load checkpoint
    model = build_model(cfg.model, train_cfg=None, test_cfg=cfg.get("test_cfg"))

    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)

    pretrained_model = torch.load(opt.pretrained_path, map_location="cpu")
    pretrained_model = pretrained_model["state_dict"]  # loaded from config
    fc_state_dict = dict()
    fc_state_dict["fc_cls.weight"] = pretrained_model["fc.weight"]
    fc_state_dict["fc_cls.bias"] = pretrained_model["fc.bias"]
    model.cls_head.load_state_dict(fc_state_dict, strict=True)
    print("loaded fc.weight and fc.bias")

    model = MMDataParallel(model, device_ids=[0])
    single_gpu_test(model, data_loader, opt.out)


def main(opt):
    assert Path(opt.pretrained_path).exists()

    cfg = Config.fromfile(opt.config)
    cfg.merge_from_dict(opt.cfg_options)

    # set cudnn benchmark
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True
    cfg.data.test.test_mode = True

    # The flag is used to register module's hooks
    cfg.setdefault("module_hooks", [])

    # build the dataloader
    print(f"extracting from {cfg.data.test.ann_file}")
    dataset = build_dataset(cfg.data.test, dict(test_mode=True))
    dataloader_setting = dict(
        videos_per_gpu=cfg.data.get("videos_per_gpu", 1),
        workers_per_gpu=cfg.data.get("workers_per_gpu", 1),
        dist=False,
        shuffle=False,
        # pin_memory=False  # not sure
    )

    dataloader_setting = dict(dataloader_setting, **cfg.data.get("test_dataloader", {}))
    data_loader = build_dataloader(dataset, **dataloader_setting)

    inference_pytorch(opt, cfg, data_loader)


if __name__ == "__main__":
    opt = parse_args()
    main(opt)
