import json
import os
from pathlib import Path

import yaml


def extract_wav(save_dir, mode, tmp_dir):
    os.makedirs(f"{tmp_dir}/{mode}/bg", exist_ok=True)
    with open(save_dir / "json" / f"bg_{mode}.json") as f:
        dict_bg = json.load(f)

    # mp4 to wav
    for name, value in dict_bg.items():
        print(value)
        path_bg = value["original_path"]
        command = f"""
        ffmpeg -i {path_bg} -vn -ac 1 -b:a 192k ./{tmp_dir}/{mode}/bg/{name}.wav
        """
        os.system(command)


if __name__ == "__main__":
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    for mode in ["train", "valid", "test"]:
        extract_wav(Path(config["save_dir"]), mode, config["tmp_dir"])
