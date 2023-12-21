from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class LinearAnnealReconWeightHook(Hook):

    def __init__(self, final_recon_weight: float = 1., stop_epoch: int = 10):
        self.final_recon_weight = final_recon_weight
        self.stop_epoch = stop_epoch

    def before_epoch(self, runner):
        recon_weight = min(self.final_recon_weight * runner.epoch / self.stop_epoch, self.final_recon_weight)
        runner.model.module.cls_head.loss_recon.loss_weight = recon_weight
