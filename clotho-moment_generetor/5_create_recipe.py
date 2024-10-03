import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml
from tqdm import tqdm


class Loader:
    def __init__(
        self,
        min_fg_db,
        max_fg_db,
        min_bg_db,
        max_bg_db,
        avg_interval,
    ):
        self.min_fg_db = min_fg_db
        self.max_fg_db = max_fg_db
        self.min_bg_db = min_bg_db
        self.max_bg_db = max_bg_db
        self.avg_interval = avg_interval
        self.qid = 0

    def create_recipe(self, path_fg, path_bg, save_dir, mode):
        with open(path_fg) as f:
            self.dict_fg = json.load(f)
        with open(path_bg) as f:
            self.dict_bg = json.load(f)

        list_recipe = []
        for name, path_bg in self.load_bg():
            data_name = f"{Path(path_bg).parent.stem}_{Path(path_bg).stem}"
            recipe = {"name": data_name, "bg": {"path": path_bg}}

            info = sf.info(path_bg)
            duration_bg = info.frames / info.samplerate

            recipe["fg"] = self.fg_sample(duration_bg)
            recipe["bg"]["dB"] = (
                random.random() * (self.max_bg_db - self.min_bg_db) + self.min_bg_db
            )

            list_recipe.append(recipe)

        with open(save_dir / "json" / f"recipe_{mode}.json", "w") as f:
            json.dump(list_recipe, f, indent=4)

    def load_bg(self):
        for name, value in self.dict_bg.items():
            for clip in tqdm(value["clips"]):
                yield name, clip

    def fg_sample(self, duration_bg):
        keys = list(self.dict_fg.keys())
        random.shuffle(keys)
        list_fg = []
        current_time = 0

        for sample_key in keys:
            current_time += np.random.exponential(self.avg_interval)
            dict_status, duration_fg = self.get_info(sample_key, current_time)
            current_time += duration_fg

            if current_time > duration_bg:
                break
            else:
                list_fg.append(dict_status)
                self.qid += 1

        list_fg.sort(key=lambda x: x["start_time"])

        return list_fg

    def get_info(self, sample_key, start_time):
        dict_status = {}

        path_fg = self.dict_fg[sample_key]["clip"]
        cap = random.sample(self.dict_fg[sample_key]["captions"], 1)[0]
        db = random.random() * (self.max_fg_db - self.min_fg_db) + self.min_fg_db

        dict_status["qid"] = self.qid
        dict_status["path"] = path_fg
        dict_status["caption"] = cap
        dict_status["dB"] = db
        dict_status["duration"] = self.dict_fg[sample_key]["duration"]
        dict_status["start_time"] = start_time

        return dict_status, dict_status["duration"]


if __name__ == "__main__":
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    save_dir = Path(config["save_dir"])
    loader = Loader(
        config["min_fg_db"],
        config["max_fg_db"],
        config["min_bg_db"],
        config["max_bg_db"],
        config["avg_interval"],
    )

    for mode in ["train", "valid", "test"]:
        fg_json = save_dir / "json" / f"fg_{mode}.json"
        bg_json = save_dir / "json" / f"bg_{mode}.json"
        loader.create_recipe(fg_json, bg_json, save_dir, mode)
