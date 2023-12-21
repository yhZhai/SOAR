from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class AnnealEDLWeightHook(Hook):

    def __init__(self, total_epochs: int = 50):
        self.total_epochs = total_epochs

    def before_epoch(self, runner):
        runner.model.module.cls_head.loss_cls.update_annealing_coef(runner.epoch, runner.max_epochs)
