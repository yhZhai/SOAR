"""
t-SNE visualization usage:
1. python experiments/t_sne_generation.py {config_path} {checkpoint_path}
    --out {output_json_path}
2. python experiments/t_sne_visualization.py -src {output_json_path} -out
    {output_figure_path} -title {figure_title}
"""

import argparse
import os
import os.path as osp
import warnings
from datetime import datetime

import mmcv
import numpy as np
import torch
import torch.distributed as dist
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.fileio.io import file_handlers
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmcv.runner.fp16_utils import wrap_fp16_model

from mmaction.apis import multi_gpu_test
from mmaction.datasets import build_dataloader, build_dataset
from mmaction.models import build_model
from mmaction.utils import register_module_hooks


def parse_args():
    parser = argparse.ArgumentParser(
        description="MMAction2 clip-level feature extraction"
    )
    parser.add_argument("config", help="test config file path")
    parser.add_argument("checkpoint", help="checkpoint file")
    parser.add_argument("--out", help="output result file in pkl/yaml/json format")
    parser.add_argument(
        "--fuse-conv-bn",
        action="store_true",
        help="Whether to fuse conv and bn, this will slightly increase"
        "the inference speed",
    )
    parser.add_argument(
        "--gpu-collect",
        action="store_true",
        help="whether to use gpu to collect results",
    )
    parser.add_argument(
        "--tmpdir",
        help="tmp directory used for collecting results from multiple "
        "workers, available when gpu-collect is not specified",
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
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    return args


def turn_off_pretrained(cfg):
    # recursively find all pretrained in the model config,
    # and set them None to avoid redundant pretrain steps for testing
    if "pretrained" in cfg:
        cfg.pretrained = None

    # recursively turn off pretrained value
    for sub_cfg in cfg.values():
        if isinstance(sub_cfg, dict):
            turn_off_pretrained(sub_cfg)


def text2tensor(text, size=256):
    nums = [ord(x) for x in text]
    assert len(nums) < size
    nums.extend([0] * (size - len(nums)))
    nums = np.array(nums, dtype=np.uint8)
    return torch.from_numpy(nums)


def tensor2text(tensor):
    # 0 may not occur in a string
    chars = [chr(x) for x in tensor if x != 0]
    return "".join(chars)


def single_gpu_test(model, data_loader):  # noqa: F811
    """Test model with a single gpu.

    This method tests model with a single gpu and
    displays test progress bar.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.

    Returns:
        list: The prediction results.
    """
    model.eval()
    results = []
    labels = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, return_latent_vector=True, **data)
        results.extend(result)
        labels.append(data["label"].item())

        # use the first key as main key to calculate the batch size
        batch_size = len(next(iter(data.values())))
        for _ in range(batch_size):
            prog_bar.update()
    return results, labels


def inference_pytorch(args, cfg, distributed, data_loader):
    """Get predictions by pytorch models."""
    # remove redundant pretrain steps for testing
    turn_off_pretrained(cfg.model)

    # build the model and load checkpoint
    model = build_model(cfg.model, train_cfg=None, test_cfg=cfg.get("test_cfg"))

    if len(cfg.module_hooks) > 0:
        register_module_hooks(model, cfg.module_hooks)

    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, args.checkpoint, map_location="cpu")

    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs, labels = single_gpu_test(model, data_loader)
    else:
        raise NotImplementedError
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
        )
        outputs = multi_gpu_test(model, data_loader, args.tmpdir, args.gpu_collect)

    return outputs, labels


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    cfg.merge_from_dict(args.cfg_options)

    # Load output_config from cfg
    output_config = cfg.get("output_config", {})
    if args.out:
        # Overwrite output_config from args.out
        output_config = Config._merge_a_into_b(dict(out=args.out), output_config)

    # set cudnn benchmark
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True
    cfg.data.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == "none":
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    rank, _ = get_dist_info()

    # The flag is used to register module's hooks
    cfg.setdefault("module_hooks", [])

    # build the dataloader
    dataset = build_dataset(cfg.data.test, dict(test_mode=True))
    dataloader_setting = dict(
        videos_per_gpu=cfg.data.get("videos_per_gpu", 1),
        workers_per_gpu=cfg.data.get("workers_per_gpu", 1),
        dist=distributed,
        shuffle=False,
    )

    dataloader_setting = dict(dataloader_setting, **cfg.data.get("test_dataloader", {}))
    data_loader = build_dataloader(dataset, **dataloader_setting)

    outputs, labels = inference_pytorch(args, cfg, distributed, data_loader)

    if rank == 0:
        if output_config.get("out", None):
            out = output_config["out"]
            print(f"\nwriting results to {out}")
            results = {}
            for i in range(len(outputs)):
                results[i] = {"feat": outputs[i], "label": labels[i]}
            dataset.dump_results(results, **output_config)


if __name__ == "__main__":
    main()
