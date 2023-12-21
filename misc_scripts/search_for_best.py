import json
import csv
from typing import List, Dict
from pathlib import Path

from rich import print


def search(root_folder: str, file_names: Dict, entry: str, output_path: str):
    root_folder = Path(root_folder)
    header = ["path", *file_names.keys()]
    with open(output_path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for folder in root_folder.iterdir():
            row = [folder] + [None] * len(file_names)
            for i, result_file in enumerate(file_names.values()):
                result_file = list(folder.glob(result_file))
                if result_file:
                    result_file = result_file[0]
                    with open(result_file, "r") as result:
                        result = json.load(result)
                        try:
                            v = result[entry]
                            row[i + 1] = v
                        except Exception as e:
                            print(
                                f"[red]could not read {entry} from {result_file}, "
                                f"because {e}[/red]"
                            )

            writer.writerow(row)


if __name__ == "__main__":
    search(
        "work_dirs",
        {
            "hmdb": "ood_latest_evidence_hmdb_result.json",
            "mit": "ood_latest_evidence_mit_result.json",
            "dense hmdb": "ood_latest_dense_evidence_hmdb_result.json",
            "dense mit": "ood_latest_dense_evidence_mit_result.json",
        },
        "overall open set auc",
        "tmp/search.csv",
    )
