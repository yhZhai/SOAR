import math

import torch
import torch.nn.functional as F

from ..builder import LOSSES
from .base import BaseWeightedLoss


def relu_evidence(y):
    return F.relu(y)


def exp_evidence(y):
    return torch.exp(torch.clamp(y, -10, 10))


def softplus_evidence(y):
    return F.softplus(y)


@LOSSES.register_module()
class EvidenceLoss(BaseWeightedLoss):
    """Evidential MSE Loss."""

    def __init__(
        self,
        num_classes,
        evidence="relu",
        loss_type="mse",
        with_kldiv=False,
        with_avuloss=False,
        disentangle=False,
        annealing_method="step",
        annealing_start=0.01,
        annealing_step=10,
        total_epochs=50,
        kldiv_weight: float = 1.0,
        avuloss_weight: float = 1.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.evidence = evidence
        self.loss_type = loss_type
        self.with_kldiv = with_kldiv
        self.with_avuloss = with_avuloss
        self.disentangle = disentangle
        self.annealing_method = annealing_method
        self.annealing_start = annealing_start
        self.annealing_step = annealing_step
        self.kldiv_weight = kldiv_weight
        self.avuloss_weight = avuloss_weight
        self.eps = 1e-10

        self.annealing_coef = self.update_annealing_coef(0, total_epochs)

    def kl_divergence(self, alpha):
        beta = torch.ones([1, self.num_classes], dtype=torch.float32).to(alpha.device)
        S_alpha = torch.sum(alpha, dim=1, keepdim=True)
        S_beta = torch.sum(beta, dim=1, keepdim=True)
        lnB = torch.lgamma(S_alpha) - torch.sum(
            torch.lgamma(alpha), dim=1, keepdim=True
        )
        lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(
            S_beta
        )

        dg0 = torch.digamma(S_alpha)
        dg1 = torch.digamma(alpha)

        kl = (
            torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
        )
        kl = kl * self.kldiv_weight
        return kl

    def loglikelihood_loss(self, y, alpha):
        S = torch.sum(alpha, dim=1, keepdim=True)
        loglikelihood_err = torch.sum((y - (alpha / S)) ** 2, dim=1, keepdim=True)
        loglikelihood_var = torch.sum(
            alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True
        )
        return loglikelihood_err, loglikelihood_var

    def mse_loss(self, y, alpha, annealing_coef):
        """Used only for loss_type == 'mse'
        y: the one-hot labels (batchsize, num_classes)
        alpha: the predictions (batchsize, num_classes)
        epoch_num: the current training epoch
        """
        losses = {}
        loglikelihood_err, loglikelihood_var = self.loglikelihood_loss(y, alpha)
        losses.update({"loss_cls": loglikelihood_err, "loss_var": loglikelihood_var})

        losses.update({"lambda": annealing_coef})
        if self.with_kldiv:
            kl_alpha = (alpha - 1) * (1 - y) + 1
            kl_div = annealing_coef * self.kl_divergence(kl_alpha)
            losses.update({"loss_kl": kl_div})

        if self.with_avuloss:
            S = torch.sum(alpha, dim=1, keepdim=True)  # Dirichlet strength
            pred_score = alpha / S
            uncertainty = self.num_classes / S
            # avu_loss = annealing_coef *
        return losses

    def ce_loss(self, target, y, alpha, annealing_coef):
        """Used only for loss_type == 'ce'
        target: the scalar labels (batchsize,)
        alpha: the predictions (batchsize, num_classes), alpha >= 1
        epoch_num: the current training epoch
        """
        losses = {}
        # (1) the classification loss term
        S = torch.sum(alpha, dim=1, keepdim=True)
        pred_score = alpha / S
        loss_cls = F.nll_loss(torch.log(pred_score), target, reduction="none")
        losses.update({"loss_cls": loss_cls})

        # (2) the likelihood variance term
        loglikelihood_var = torch.sum(
            alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True
        )
        losses.update({"loss_var": loglikelihood_var})

        # (3) the KL divergence term
        kl_alpha = (alpha - 1) * (1 - y) + 1
        kl_div = annealing_coef * self.kl_divergence(kl_alpha)
        losses.update({"loss_kl": kl_div, "lambda": annealing_coef})
        return losses

    def edl_loss(self, func, y, alpha, annealing_coef, target):
        """Used for both loss_type == 'log' and loss_type == 'digamma'
        func: function handler (torch.log, or torch.digamma)
        y: the one-hot labels (batchsize, num_classes)
        alpha: the predictions (batchsize, num_classes)
        epoch_num: the current training epoch
        """
        losses = {}
        S = torch.sum(alpha, dim=1, keepdim=True)
        A = torch.sum(y * (func(S) - func(alpha)), dim=1, keepdim=True)
        losses.update({"loss_cls": A})

        losses.update({"lambda": annealing_coef})
        if self.with_kldiv:
            kl_alpha = (alpha - 1) * (1 - y) + 1
            kl_div = annealing_coef * self.kl_divergence(kl_alpha)
            losses.update({"loss_kl": kl_div})

        if self.with_avuloss:
            pred_scores, pred_cls = torch.max(alpha / S, 1, keepdim=True)
            uncertainty = self.num_classes / S
            acc_match = torch.reshape(
                torch.eq(pred_cls, target.unsqueeze(1)).float(), (-1, 1)
            )
            if self.disentangle:
                acc_uncertain = -torch.log(pred_scores * (1 - uncertainty) + self.eps)
                inacc_certain = -torch.log((1 - pred_scores) * uncertainty + self.eps)
            else:
                acc_uncertain = -pred_scores * torch.log(1 - uncertainty + self.eps)
                inacc_certain = -(1 - pred_scores) * torch.log(uncertainty + self.eps)
            avu_loss = (
                annealing_coef * acc_match * acc_uncertain
                + (1 - annealing_coef) * (1 - acc_match) * inacc_certain
            )
            avu_loss = avu_loss * self.avuloss_weight
            losses.update({"loss_avu": avu_loss})
        return losses

    def update_annealing_coef(self, epoch, total_epoch):
        # annealing coefficient
        if self.annealing_method == "step":
            annealing_coef = min(1, epoch / self.annealing_step)
        elif self.annealing_method == "exp":
            annealing_coef = self.annealing_start * math.exp(
                -math.log(self.annealing_start) / total_epoch * epoch
            )
        else:
            raise NotImplementedError

        self.annealing_coef = torch.tensor(annealing_coef, dtype=torch.float32)

    def _forward(self, output, target, **kwargs):
        """Forward function.
        Args:
            output (torch.Tensor): The class score (before softmax).
            target (torch.Tensor): The ground truth label.
        Returns:
            torch.Tensor: The returned EvidenceLoss loss.
        """
        # get evidence
        if self.evidence == "relu":
            evidence = relu_evidence(output)
        elif self.evidence == "exp":
            evidence = exp_evidence(output)
        elif self.evidence == "softplus":
            evidence = softplus_evidence(output)
        else:
            raise NotImplementedError
        alpha = evidence + 1

        # one-hot embedding for the target
        y = F.one_hot(target, self.num_classes).to(output.device)

        # compute the EDL loss
        self.annealing_coef = self.annealing_coef.to(output.device)
        if self.loss_type == "mse":
            results = self.mse_loss(y, alpha, self.annealing_coef)
        elif self.loss_type == "log":
            results = self.edl_loss(torch.log, y, alpha, self.annealing_coef, target)
        elif self.loss_type == "digamma":
            results = self.edl_loss(
                torch.digamma, y, alpha, self.annealing_coef, target
            )
        elif self.loss_type == "cross_entropy":
            results = self.ce_loss(target, y, alpha, self.annealing_coef)
        else:
            raise NotImplementedError

        # compute uncertainty and evidence
        _, preds = torch.max(output, 1)
        match = torch.reshape(torch.eq(preds, target).float(), (-1, 1))
        uncertainty = self.num_classes / torch.sum(alpha, dim=1, keepdim=True)
        total_evidence = torch.sum(evidence, 1, keepdim=True)
        evidence_succ = torch.sum(total_evidence * match) / torch.sum(match + 1e-20)
        evidence_fail = torch.sum(total_evidence * (1 - match)) / (
            torch.sum(torch.abs(1 - match)) + 1e-20
        )
        results.update(
            {
                "uncertainty": uncertainty,
                "evidence_succ": evidence_succ,
                "evidence_fail": evidence_fail,
            }
        )

        return results
