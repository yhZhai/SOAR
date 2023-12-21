from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class LinearAnnealKLWeightHook(Hook):

    def __init__(self, final_kl_weight: float = 1., stop_epoch: int = 10):
        self.final_kl_weight = final_kl_weight
        self.stop_epoch = stop_epoch

    def before_epoch(self, runner):
        kl_weight = min(self.final_kl_weight * runner.epoch / runner.max_epochs, self.final_kl_weight)
        runner.model.module.cls_head.loss_kld.loss_weight = kl_weight
