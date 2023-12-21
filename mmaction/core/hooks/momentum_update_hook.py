import torch
from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class MomentumUpdateHook(Hook):

    def __init__(self, momentum: float = 0.999) -> None:
        assert 1 >= momentum >= 0, "Momentum should be in [0, 1]"
        self.momentum = momentum

    def after_iter(self, runner):
        with torch.no_grad():
            runner.model.module.cls_head.cluster_centers.data = runner.model.module.cls_head.prev_cluster_centers.data * \
                self.momentum + runner.model.module.cls_head.cluster_centers.data * \
                (1 - self.momentum)
            runner.model.module.cls_head.prev_cluster_centers.data = runner.model.module.cls_head.cluster_centers.data
