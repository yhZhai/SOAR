from typing import Dict

import torch
import torch.nn.functional as F


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.sum = 0
        self.avg = 0
        self.val = 0
        self.count = 0

    def reset(self):
        self.sum = 0
        self.avg = 0
        self.val = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum = self.sum + val * n
        self.count = self.count + n
        self.avg = self.sum / self.count

    def __str__(self):
        return f"{self.avg: .7f}"


def get_recon_error_uncertainty(outputs, recon_loss_fn=F.mse_loss):
    recon_error = recon_loss_fn(outputs["recon"], outputs["gt_recon"])
    recon_error = recon_error.unsqueeze(0).unsqueeze(0)
    uncertainty = recon_error.detach().cpu().numpy()  # b
    raw_score = outputs["cls_score"]
    score = raw_score.softmax(dim=-1).detach().cpu().numpy()  # b, c
    raw_score = raw_score.detach().cpu().numpy()  # b, c
    return score, uncertainty


def get_evidential_learning_uncertainty(
    get_evidence_fn, num_classes, outputs, tgt_key="cls_score"
):
    cls_score = outputs[tgt_key]
    evidence = get_evidence_fn(cls_score)
    alpha = evidence + 1
    uncertainty = num_classes / torch.sum(alpha, dim=1)
    raw_score = cls_score
    score = alpha / torch.sum(alpha, dim=1, keepdim=True)
    raw_score = raw_score.detach().cpu().numpy()  # b, c
    score = score.detach().cpu().numpy()  # b, c
    uncertainty = uncertainty.detach().cpu().numpy()  # b
    return score, uncertainty


def turn_off_pretrained(cfg):
    # recursively find all pretrained in the model config,
    # and set them None to avoid redundant pretrain steps for testing
    if "pretrained" in cfg:
        cfg.pretrained = None

    # recursively turn off pretrained value
    for sub_cfg in cfg.values():
        if isinstance(sub_cfg, dict):
            turn_off_pretrained(sub_cfg)


class UnNormalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


def minmax_normalization(x):
    x = (x - x.min()) / (x.max() - x.min())
    return x


def apply_dropout(m):
    if isinstance(m, torch.nn.Dropout):
        m.train()
        print(f"activating dropout {m}")


def update_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def preprocess(data: Dict, key: str, proc: str):
    if proc == "none":
        return data

    if proc == "freeze":
        t = data[key].shape[3]
        data["imgs"] = data["imgs"][:, :, :, 0, None, ...].repeat(1, 1, 1, t, 1, 1)
    elif proc == "shuffle":
        t = data["imgs"].shape[3]
        data["imgs"] = data["imgs"][:, :, :, torch.randperm(t), ...]
    else:
        raise NotImplementedError(f"unsupported preprocess {proc}")

    return data
