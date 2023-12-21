from typing import Optional

from rich import print

from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class LinearAnnealGradRevAlphaHook(Hook):
    def __init__(self, target: str = "alpha"):
        assert target in ["alpha", "weight"]
        self.target = target
        print(f"[red]using linear anneal hook on {self.target}[/red]")

        self.num_epoch: Optional[int] = None
        self.final_recon_grad_rev_alpha: Optional[float] = None
        self.final_scene_grad_rev_alpha: Optional[float] = None
        self.final_recon_loss_weight: Optional[float] = None
        self.final_scene_cls_loss_weight: Optional[float] = None

    def before_run(self, runner):
        self.num_epoch = runner.max_epochs
        self.final_scene_grad_rev_alpha = runner.model.module.cls_head.fc_scene_cls[0].alpha.item()
        self.final_recon_grad_rev_alpha = runner.model.module.cls_head.pre_decoder.alpha.item()
        self.final_recon_loss_weight = runner.model.module.cls_head.loss_recon.loss_weight
        self.final_scene_cls_loss_weight = runner.model.module.cls_head.loss_debias.loss_weight

    def before_epoch(self, runner):
        current_epoch = runner.epoch
        if self.target == "alpha":
            scene_grad_rev_alpha = current_epoch / self.num_epoch * self.final_scene_grad_rev_alpha
            recon_grad_rev_alpha = current_epoch / self.num_epoch * self.final_recon_grad_rev_alpha
            runner.model.module.cls_head.fc_scene_cls[0].update_alpha(scene_grad_rev_alpha)
            runner.model.module.cls_head.pre_decoder.update_alpha(recon_grad_rev_alpha)
        elif self.target == "weight":
            recon_loss_weight = current_epoch / self.num_epoch * self.final_recon_loss_weight
            scene_cls_loss_weight = current_epoch / self.num_epoch * self.final_scene_cls_loss_weight
            runner.model.module.cls_head.loss_recon.loss_weight = recon_loss_weight
            runner.model.module.cls_head.loss_debias.loss_weight = scene_cls_loss_weight
        else:
            raise NotImplementedError(f"not support target {self.target}")
