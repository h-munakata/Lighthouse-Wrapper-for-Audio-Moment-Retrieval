import csv
import json
import os
from pathlib import Path

import yaml


def collect_fg(save_dir, root_clotho, mode, clotho_mode):
    dict_fg = {}
    # load captions
    with open(root_clotho / f"clotho_captions_{clotho_mode}.csv") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            key = row[0][:-4]
            dict_fg[key] = {"captions": [cap for n, cap in enumerate(row[1:])]}

    # load wavs
    for key in dict_fg.keys():
        originak_path = root_clotho / clotho_mode / f"{key}.wav"
        dict_fg[key].update({"original_path": str(originak_path)})

    print(f"Number of foreground clips in {mode}: {len(dict_fg)}")

    # save data
    with open(save_dir / "json" / f"fg_{mode}.json", "w") as f:
        json.dump(dict_fg, f, indent=4)


def collect_bg(save_dir, root_wwt, split_ratio):
    split_ratio = [r / sum(split_ratio) for r in split_ratio]
    list_bg = list(root_wwt.glob("*.mp4"))
    num_tr = round(len(list_bg) * split_ratio[0])
    num_vl = round(len(list_bg) * split_ratio[1])

    split_list_bg = [
        list_bg[:num_tr],
        list_bg[num_tr : num_tr + num_vl],
        list_bg[num_tr + num_vl :],
    ]

    for mode, _list_bg in zip(["train", "valid", "test"], split_list_bg):
        dict_bg = {}
        for path_bg in _list_bg:
            dict_bg[path_bg.stem] = {"original_path": str(path_bg)}

        print(f"Number of background clips in {mode}: {len(dict_bg)}")

        with open(save_dir / "json" / f"bg_{mode}.json", "w") as f:
            json.dump(dict_bg, f, indent=4)


if __name__ == "__main__":
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    save_dir = Path(config["save_dir"])

    os.makedirs(save_dir / "json", exist_ok=True)

    for mode, clotho_mode in zip(
        ["train", "valid", "test"], ["development", "validation", "evaluation"]
    ):
        collect_fg(save_dir, Path(config["root_clotho"]), mode, clotho_mode)

    collect_bg(save_dir, Path(config["root_wwt"]), config["split_ratio"])
