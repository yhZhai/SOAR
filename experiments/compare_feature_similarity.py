# https://colab.research.google.com/github/google-research/google-research/blob/master/representation_similarity/Demo.ipynb#scrollTo=45qb6zdSsHj6

import argparse
from pathlib import Path
import json
import copy

import tqdm
from rich import print
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import torch
import torch.nn as nn
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
    parser.add_argument("--output", help="output folder name", default="feature")
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


def to_numpy(x):
    """convert Pytorch tensor to numpy array"""
    return x.clone().detach().cpu().numpy()


class HSIC(nn.Module):
    """Base class for the finite sample estimator of Hilbert-Schmidt Independence Criterion (HSIC)
    ..math:: HSIC (X, Y) := || C_{x, y} ||^2_{HS}, where HSIC (X, Y) = 0 iif X and Y are independent.
    Empirically, we use the finite sample estimator of HSIC (with m observations) by,
    (1) biased estimator (HSIC_0)
        Gretton, Arthur, et al. "Measuring statistical dependence with Hilbert-Schmidt norms." 2005.
        :math: (m - 1)^2 tr KHLH.
        where K_{ij} = kernel_x (x_i, x_j), L_{ij} = kernel_y (y_i, y_j), H = 1 - m^{-1} 1 1 (Hence, K, L, H are m by m matrices).
    (2) unbiased estimator (HSIC_1)
        Song, Le, et al. "Feature selection via dependence maximization." 2012.
        :math: \frac{1}{m (m - 3)} \bigg[ tr (\tilde K \tilde L) + \frac{1^\top \tilde K 1 1^\top \tilde L 1}{(m-1)(m-2)} - \frac{2}{m-2} 1^\top \tilde K \tilde L 1 \bigg].
        where \tilde K and \tilde L are related to K and L by the diagonal entries of \tilde K_{ij} and \tilde L_{ij} are set to zero.
    Parameters
    ----------
    sigma_x : float
        the kernel size of the kernel function for X.
    sigma_y : float
        the kernel size of the kernel function for Y.
    algorithm: str ('unbiased' / 'biased')
        the algorithm for the finite sample estimator. 'unbiased' is used for our paper.
    reduction: not used (for compatibility with other losses).
    """

    def __init__(self, sigma_x, sigma_y=None, algorithm="unbiased", reduction=None):
        super(HSIC, self).__init__()

        if sigma_y is None:
            sigma_y = sigma_x

        self.sigma_x = sigma_x
        self.sigma_y = sigma_y

        if algorithm == "biased":
            self.estimator = self.biased_estimator
        elif algorithm == "unbiased":
            self.estimator = self.unbiased_estimator
        else:
            raise ValueError("invalid estimator: {}".format(algorithm))

    def _kernel_x(self, X):
        raise NotImplementedError

    def _kernel_y(self, Y):
        raise NotImplementedError

    def biased_estimator(self, input1, input2):
        """Biased estimator of Hilbert-Schmidt Independence Criterion
        Gretton, Arthur, et al. "Measuring statistical dependence with Hilbert-Schmidt norms." 2005.
        """
        K = self._kernel_x(input1)
        L = self._kernel_y(input2)

        KH = K - K.mean(0, keepdim=True)
        LH = L - L.mean(0, keepdim=True)

        N = len(input1)

        return torch.trace(KH @ LH / (N - 1) ** 2)

    def unbiased_estimator(self, input1, input2):
        """Unbiased estimator of Hilbert-Schmidt Independence Criterion
        Song, Le, et al. "Feature selection via dependence maximization." 2012.
        """
        kernel_XX = self._kernel_x(input1)
        kernel_YY = self._kernel_y(input2)

        tK = kernel_XX - torch.diag(kernel_XX)
        tL = kernel_YY - torch.diag(kernel_YY)

        N = len(input1)

        hsic = (
            torch.trace(tK @ tL)
            + (torch.sum(tK) * torch.sum(tL) / (N - 1) / (N - 2))
            - (2 * torch.sum(tK, 0).dot(torch.sum(tL, 0)) / (N - 2))
        )

        return hsic / (N * (N - 3))

    def forward(self, input1, input2, **kwargs):
        return self.estimator(input1, input2)


class RbfHSIC(HSIC):
    """Radial Basis Function (RBF) kernel HSIC implementation."""

    def _kernel(self, X, sigma):
        X = X.view(len(X), -1)
        XX = X @ X.t()
        X_sqnorms = torch.diag(XX)
        X_L2 = -2 * XX + X_sqnorms.unsqueeze(1) + X_sqnorms.unsqueeze(0)
        gamma = 1 / (2 * sigma**2)

        kernel_XX = torch.exp(-gamma * X_L2)
        return kernel_XX

    def _kernel_x(self, X):
        return self._kernel(X, self.sigma_x)

    def _kernel_y(self, Y):
        return self._kernel(Y, self.sigma_y)


def gram_linear(x):
    """Compute Gram (kernel) matrix for a linear kernel.

    Args:
      x: A num_examples x num_features matrix of features.

    Returns:
      A num_examples x num_examples Gram matrix of examples.
    """
    return x.dot(x.T)


def gram_rbf(x, threshold=1.0):
    """Compute Gram (kernel) matrix for an RBF kernel.

    Args:
      x: A num_examples x num_features matrix of features.
      threshold: Fraction of median Euclidean distance to use as RBF kernel
        bandwidth. (This is the heuristic we use in the paper. There are other
        possible ways to set the bandwidth; we didn't try them.)

    Returns:
      A num_examples x num_examples Gram matrix of examples.
    """
    dot_products = x.dot(x.T)
    sq_norms = np.diag(dot_products)
    sq_distances = -2 * dot_products + sq_norms[:, None] + sq_norms[None, :]
    sq_median_distance = np.median(sq_distances)
    return np.exp(-sq_distances / (2 * threshold**2 * sq_median_distance))


def center_gram(gram, unbiased=False):
    """Center a symmetric Gram matrix.

    This is equvialent to centering the (possibly infinite-dimensional) features
    induced by the kernel before computing the Gram matrix.

    Args:
      gram: A num_examples x num_examples symmetric matrix.
      unbiased: Whether to adjust the Gram matrix in order to compute an unbiased
        estimate of HSIC. Note that this estimator may be negative.

    Returns:
      A symmetric matrix with centered columns and rows.
    """
    if not np.allclose(gram, gram.T):
        raise ValueError("Input must be a symmetric matrix.")
    gram = gram.copy()

    if unbiased:
        # This formulation of the U-statistic, from Szekely, G. J., & Rizzo, M.
        # L. (2014). Partial distance correlation with methods for dissimilarities.
        # The Annals of Statistics, 42(6), 2382-2412, seems to be more numerically
        # stable than the alternative from Song et al. (2007).
        n = gram.shape[0]
        np.fill_diagonal(gram, 0)
        means = np.sum(gram, 0, dtype=np.float64) / (n - 2)
        means -= np.sum(means) / (2 * (n - 1))
        gram -= means[:, None]
        gram -= means[None, :]
        np.fill_diagonal(gram, 0)
    else:
        means = np.mean(gram, 0, dtype=np.float64)
        means -= np.mean(means) / 2
        gram -= means[:, None]
        gram -= means[None, :]

    return gram


def cka(gram_x, gram_y, debiased=False):
    """Compute CKA.

    Args:
      gram_x: A num_examples x num_examples Gram matrix.
      gram_y: A num_examples x num_examples Gram matrix.
      debiased: Use unbiased estimator of HSIC. CKA may still be biased.

    Returns:
      The value of CKA between X and Y.
    """
    gram_x = center_gram(gram_x, unbiased=debiased)
    gram_y = center_gram(gram_y, unbiased=debiased)

    # Note: To obtain HSIC, this should be divided by (n-1)**2 (biased variant) or
    # n*(n-3) (unbiased variant), but this cancels for CKA.
    scaled_hsic = gram_x.ravel().dot(gram_y.ravel())

    normalization_x = np.linalg.norm(gram_x)
    normalization_y = np.linalg.norm(gram_y)
    return scaled_hsic / (normalization_x * normalization_y)


def _debiased_dot_product_similarity_helper(
    xty, sum_squared_rows_x, sum_squared_rows_y, squared_norm_x, squared_norm_y, n
):
    """Helper for computing debiased dot product similarity (i.e. linear HSIC)."""
    # This formula can be derived by manipulating the unbiased estimator from
    # Song et al. (2007).
    return (
        xty
        - n / (n - 2.0) * sum_squared_rows_x.dot(sum_squared_rows_y)
        + squared_norm_x * squared_norm_y / ((n - 1) * (n - 2))
    )


def feature_space_linear_cka(features_x, features_y, debiased=False):
    """Compute CKA with a linear kernel, in feature space.

    This is typically faster than computing the Gram matrix when there are fewer
    features than examples.

    Args:
      features_x: A num_examples x num_features matrix of features.
      features_y: A num_examples x num_features matrix of features.
      debiased: Use unbiased estimator of dot product similarity. CKA may still be
        biased. Note that this estimator may be negative.

    Returns:
      The value of CKA between X and Y.
    """
    features_x = features_x - np.mean(features_x, 0, keepdims=True)
    features_y = features_y - np.mean(features_y, 0, keepdims=True)

    dot_product_similarity = np.linalg.norm(features_x.T.dot(features_y)) ** 2
    normalization_x = np.linalg.norm(features_x.T.dot(features_x))
    normalization_y = np.linalg.norm(features_y.T.dot(features_y))

    if debiased:
        n = features_x.shape[0]
        # Equivalent to np.sum(features_x ** 2, 1) but avoids an intermediate array.
        sum_squared_rows_x = np.einsum("ij,ij->i", features_x, features_x)
        sum_squared_rows_y = np.einsum("ij,ij->i", features_y, features_y)
        squared_norm_x = np.sum(sum_squared_rows_x)
        squared_norm_y = np.sum(sum_squared_rows_y)

        dot_product_similarity = _debiased_dot_product_similarity_helper(
            dot_product_similarity,
            sum_squared_rows_x,
            sum_squared_rows_y,
            squared_norm_x,
            squared_norm_y,
            n,
        )
        normalization_x = np.sqrt(
            _debiased_dot_product_similarity_helper(
                normalization_x**2,
                sum_squared_rows_x,
                sum_squared_rows_x,
                squared_norm_x,
                squared_norm_x,
                n,
            )
        )
        normalization_y = np.sqrt(
            _debiased_dot_product_similarity_helper(
                normalization_y**2,
                sum_squared_rows_y,
                sum_squared_rows_y,
                squared_norm_y,
                squared_norm_y,
                n,
            )
        )

    return dot_product_similarity / (normalization_x * normalization_y)


def get_results(
    model,
    data_loader,
    folder: Path,
):
    hsic_model = RbfHSIC(np.sqrt(2048)).cuda()

    pbar = tqdm.tqdm(enumerate(data_loader), total=len(data_loader))
    model.eval()
    scene_feature_list = []
    action_feature_list = []
    for i, data in pbar:

        if "filename" in data["img_metas"].data[0][0].keys():
            frame_dir = Path(data["img_metas"].data[0][0]["filename"]).stem
        elif "frame_dir" in data["img_metas"].data[0][0].keys():
            frame_dir = Path(data["img_metas"].data[0][0]["frame_dir"]).stem
        else:
            raise NotImplementedError

        scene_feature = data["scene_feature"]
        action_feature_path = (folder / frame_dir).with_suffix(".npy")
        if not action_feature_path.exists():
            with torch.no_grad():
                outputs = model(
                    return_loss=False, **data, return_dict=True, get_cams=True
                )
                action_feature = outputs["feature"]
                action_feature = action_feature.detach().cpu().numpy()
                np.save(action_feature_path, action_feature)
        else:
            action_feature = np.load(action_feature_path)

        scene_feature_list.append(scene_feature)
        action_feature_list.append(action_feature)

    scene_feature = np.concatenate(scene_feature_list, axis=0)
    action_feature = np.concatenate(action_feature_list, axis=0)
    linear_cka = feature_space_linear_cka(scene_feature, action_feature)
    scene_feature = torch.tensor(scene_feature).cuda()
    action_feature = torch.tensor(action_feature).cuda()
    hsic = hsic_model(scene_feature, action_feature)
    return {"linear cka": linear_cka, "hsic": hsic.item()}


def run_inference(model, datalist_file, cfg, opt, subfolder: Path):
    subfolder.mkdir(parents=True, exist_ok=True)

    # switch config for different dataset
    cfg.data.test.ann_file = datalist_file
    if "mit" in datalist_file:
        cfg.data.test.type = "VideoSceneFeatureDataset"
        cfg.data.test.data_prefix = (Path(datalist_file).parent / "videos").as_posix()
        cfg.data.test.pipeline.insert(0, dict(type="DecordInit"))
        cfg.data.test.pipeline[1] = dict(
            type="SampleFrames",
            clip_len=32,
            frame_interval=2,
            num_clips=1,
            test_mode=True,
        )
        cfg.data.test.pipeline[2] = dict(type="DecordDecode")
        cfg.data.test.pipeline[4] = dict(type="CenterCrop", crop_size=256)
        cfg.data.test.pipeline[-2]["meta_keys"] = ["filename"]
    else:
        cfg.data.test.data_prefix = (
            Path(datalist_file).parent / "rawframes"
        ).as_posix()
        cfg.data.test.pipeline[0] = dict(
            type="SampleFrames",
            clip_len=32,
            frame_interval=2,
            num_clips=1,
            test_mode=True,
        )
        cfg.data.test.pipeline[3] = dict(type="CenterCrop", crop_size=256)

    if "mit" in datalist_file:
        cfg.data.test.scene_feature_root = "data/mit_scene_feature"
        cfg.data.test.scene_pred_root = "data/mit_scene_pred"
    elif "hmdb" in datalist_file:
        cfg.data.test.scene_feature_root = "data/hmdb51_scene_feature"
        cfg.data.test.scene_pred_root = "data/hmdb51_scene_pred"
    elif "ucf" in datalist_file:
        cfg.data.test.scene_feature_root = "data/ucf101_scene_feature"
        cfg.data.test.scene_pred_root = "data/ucf101_scene_pred"
    else:
        raise NotImplementedError(f"not supported datalist {datalist_file}")

    # build the dataloader
    dataset = build_dataset(cfg.data.test, dict(test_mode=True))
    dataloader_setting = dict(
        videos_per_gpu=cfg.data.get("videos_per_gpu", 1),
        workers_per_gpu=cfg.data.get("workers_per_gpu", 2),
        dist=False,
        shuffle=False,
        pin_memory=False,
    )
    dataloader_setting = dict(dataloader_setting, **cfg.data.get("test_dataloader", {}))
    data_loader = build_dataloader(dataset, **dataloader_setting)

    results = get_results(
        model,
        data_loader,
        subfolder,
    )
    return results


def main(opt):

    torch.multiprocessing.set_sharing_strategy("file_system")

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

    train_results, ood_results, ind_results = None, None, None
    ood_name = Path(opt.ood_data).name.split("_")[0]
    # # run inference (train set)
    # train_results = run_inference(model, opt.train_data, cfg, opt, vis_path / "train")
    # # run inference (OOD)
    # ood_results = run_inference(model, opt.ood_data, copy.deepcopy(cfg), opt, vis_path / ood_name)
    # run inference (IND)
    ind_results = run_inference(
        model, opt.ind_data, copy.deepcopy(cfg), opt, vis_path / "ind_test"
    )

    result = {
        "train results": train_results,
        f"{ood_name} results": ood_results,
        "ind results": ind_results,
    }
    print(result)
    output_path = vis_path / "feature_similarity.json"
    if Path(output_path).exists():
        with open(output_path, "r") as f:
            old_result = json.load(f)
            result = result.update(old_result)
    with open(output_path, "w") as f:
        json.dump(result, f)


if __name__ == "__main__":

    opt = parse_args()
    main(opt)
