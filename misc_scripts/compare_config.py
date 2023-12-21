from collections.abc import MutableMapping
from typing import Dict
import shutil
from pathlib import Path
import argparse
import importlib

import prettytable as pt
from mmcv import Config


def parse_args():
    parser = argparse.ArgumentParser("config files comparison")
    parser.add_argument("config1", type=str)
    parser.add_argument("config2", type=str)
    parser.add_argument("--tmp_cfg_tmpl", type=str, default="tmp/tmp_config{}.py")
    opt = parser.parse_args()
    return opt


def flatten_dict(d, parent_key="", sep=".") -> Dict:
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def main(
    opt,
    ignore_params=[
        "work_dir",
        "log_config",
    ],
    omit_keys=[],
):
    config1_path = Path(opt.config1)
    config2_path = Path(opt.config2)
    assert config1_path.exists()
    assert config2_path.exists()

    config1_params = Config.fromfile(opt.config1)
    config2_params = Config.fromfile(opt.config2)
    for k in ignore_params:
        config1_params.pop(k)
        config2_params.pop(k)
    config1_params = flatten_dict(config1_params)
    config2_params = flatten_dict(config2_params)
    k1 = sorted(list(config1_params.keys()))
    k2 = sorted(list(config2_params.keys()))

    tb = pt.PrettyTable()
    tb.field_names = ["key", "config1", "config2"]

    for k in k1:
        if k in k2 and not k in omit_keys:
            if isinstance(config1_params[k], list):
                if len(config1_params[k]) != len(config2_params[k]):
                    tb.add_row([f"len({k})", len(config1_params[k]), len(config2_params[k])])
                else:
                    for i, (v1, v2) in enumerate(zip(config1_params[k], config2_params[k])):
                        if v1 != v2:
                            tb.add_row([f"{k}[{i}]", v1, v2])
            elif isinstance(config1_params[k], dict):
                pass
            else:
                if config1_params[k] != config2_params[k]:
                    tb.add_row([k, config1_params[k], config2_params[k]])

    file1_new_key = list(set(k1).difference(set(k2)))
    for new_key in file1_new_key:
        tb.add_row([new_key, f"{config1_params[new_key]} (NEWKEY)", "NONE"])

    file2_new_key = list(set(k2).difference(set(k1)))
    for new_key in file2_new_key:
        tb.add_row([new_key, "NONE", f"{config2_params[new_key]} (NEWKEY)"])

    print(f"config1: {opt.config1}")
    print(f"config2: {opt.config2}")
    print(tb)


if __name__ == "__main__":
    opt = parse_args()
    main(opt)
