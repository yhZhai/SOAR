import copy
from typing import List
from collections import OrderedDict

import torch


def rename(src, tgt):
    checkpoint = torch.load(src, map_location="cpu")

    # new_state_dict = OrderedDict()
    # for k, v in checkpoint['state_dict'].items():
    #     new_k: List = k.split('.')
    #     new_k.insert(1, 'backbone')
    #     new_k = '.'.join(new_k)
    #     new_state_dict[new_k] = v

    state_dict = {
        str.replace(k, "module.", ""): v for k, v in checkpoint["state_dict"].items()
    }
    new_checkpoint = copy.deepcopy(checkpoint)
    new_checkpoint["state_dict"] = state_dict
    torch.save(new_checkpoint, tgt)


if __name__ == "__main__":
    rename("pretrained/resnet50_places365.pth.tar", "pretrained/resnet50_places365.pth")
