import math

from matplotlib import pyplot as plt
from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class CosineAnnealAdvReconWeightHook(Hook):

    def __init__(self, peak_recon_alpha: float = -1.0,
                 peak_rev_alpha: float = 1.0, stop_epoch: int = 50,
                 freeze_decoder: bool = False, anneal_recon: bool = True,
                 anneal_rev: bool = True):
        """
        if anneal_recon == True and anneal_rev == True:
            rev_alpha ↑
                      |        /----------------  ← peak_rev_alpha
                      |       /
                      |      /
                      |     /
                    0 ----------------------------→ epoch
                      |   /     ↑ stop_epoch
                      |  / ↑ freeze decoder at this point
                      | /
                      |/     ← peak_recon_alpha

        if anneal_recon == False:
            rev_alpha ↑
                      |        /----------------  ← peak_rev_alpha
                      |       /
                      |      /
                      |     /
                    0 ----------------------------→ epoch
                      |         ↑ stop_epoch
                      |    ↑ freeze decoder at this point
                      |  
                      |----- ← peak_recon_alpha
        
        if anneal_rev == False:
            rev_alpha ↑
                      |     --------------------  ← peak_rev_alpha
                      |        
                      |       
                      |      
                    0 ----------------------------→ epoch
                      |   /     ↑ stop_epoch
                      |  / ↑ freeze decoder at this point
                      | /
                      |/     ← peak_recon_alpha

        """
        assert peak_recon_alpha <= 0
        assert peak_rev_alpha >= 0
        self.peak_recon_alpha = peak_recon_alpha
        self.peak_rev_alpha = peak_rev_alpha
        self.stop_epoch = stop_epoch
        self.freeze_decoder = freeze_decoder
        self.anneal_recon = anneal_recon
        self.anneal_rev = anneal_rev

    def before_epoch(self, runner):
        epoch = runner.epoch
        alpha = self._get_alpha(epoch)
        if alpha > 0 and self.freeze_decoder:
            # do the freeze
            runner.model.module.cls_head.decoder.eval()
            runner.model.module.cls_head.decoder.requires_grad = False
            self.freeze_decoder = False

        runner.model.module.cls_head.pre_decoder.update_alpha(alpha)

    def _get_alpha(self, epoch):
        if epoch < self.stop_epoch:  # still raising
            rev_alpha = math.sin((epoch - self.stop_epoch / 2) / self.stop_epoch * math.pi)
            if rev_alpha <= 0:
                rev_alpha = rev_alpha * abs(self.peak_recon_alpha)
            elif rev_alpha > 0:
                rev_alpha = rev_alpha * abs(self.peak_rev_alpha)

            if rev_alpha <= 0 and (not self.anneal_recon):  
                return self.peak_recon_alpha
            elif rev_alpha > 0 and (not self.anneal_rev):
                return self.peak_rev_alpha

            return rev_alpha
        else:
            return self.peak_rev_alpha


if __name__ == "__main__":
    num_epoch = 50
    hook = CosineAnnealAdvReconWeightHook(
        peak_recon_alpha=-0.02,
        peak_rev_alpha=0.2,
        stop_epoch=30,
        freeze_decoder=True,
        anneal_recon=True,
        anneal_rev=False
    )
    k, v = [], []
    for i in range(num_epoch):
        k.append(i)
        v.append(hook._get_alpha(i))
    plt.plot(k, v)
    plt.xlabel("epoch")
    plt.ylabel("alpha")
    plt.savefig("tmp/anneal.png")
