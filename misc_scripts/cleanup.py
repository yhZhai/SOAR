import argparse
from pathlib import Path
import shutil

from rich import print


def clean(src_path="work_dirs", tgt_path="obsolete_work_dirs"):
    path = Path(src_path)
    folders = path.glob("*")
    folders = [folder for folder in folders if folder.is_dir()]
    for folder in folders:
        if "debug" in folder.as_posix():
            print("deleting", folder)
            shutil.rmtree(folder.as_posix())
            continue
        num_checkpoint = len(list(folder.glob("*.pth")))
        if num_checkpoint <= 2:  # the first eval and latest.pth
            print("moving", folder, "to obsolete/")
            shutil.move(
                folder.as_posix(), (Path(tgt_path) / Path(folder.name)).as_posix()
            )


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('')
    clean()
