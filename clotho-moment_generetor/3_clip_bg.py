import json
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml
from tqdm import tqdm


def preprocess(save_dir, mode, tmp_dir, clip_duration, clip_interval):
    dict_clip_bg = {}
    path_wav = tmp_dir / mode / "bg"
    for fname in path_wav.glob("*.wav"):
        os.makedirs(path_wav / fname.stem, exist_ok=True)
        dict_clip_bg[fname.stem] = {"original_path": str(fname), "clips": []}

        s, sr = sf.read(fname)
        for start_sample in tqdm(np.arange(0, len(s), clip_interval * sr)):
            end_sample = start_sample + clip_duration * sr
            if end_sample > len(s):
                break
            start_sample = int(start_sample)
            end_sample = int(end_sample)

            _s = s[start_sample:end_sample]
            start_sec, end_sec = round(start_sample / sr, 1), round(end_sample / sr, 1)

            save_path = str(path_wav / fname.stem / f"{start_sec}_{end_sec}.wav")

            sf.write(save_path, _s, sr)
            dict_clip_bg[fname.stem]["clips"].append(save_path)

    with open(save_dir / "json" / f"bg_{mode}.json", "w") as f:
        json.dump(dict_clip_bg, f, indent=2)


if __name__ == "__main__":
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    save_dir = Path(config["save_dir"])

    for mode in ["train", "valid", "test"]:
        preprocess(
            Path(config["save_dir"]),
            mode,
            Path(config["tmp_dir"]),
            config["clip_duration"],
            config["clip_interval"],
        )
