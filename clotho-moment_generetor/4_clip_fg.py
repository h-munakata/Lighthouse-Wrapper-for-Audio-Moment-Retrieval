import json
from pathlib import Path

import soundfile as sf
import yaml
from tqdm import tqdm


def preprocess(save_dir, mode, tmp_dir, clip_db):
    with open(save_dir / "json" / f"fg_{mode}.json") as f:
        dict_fg = json.load(f)

    dir_wav = Path(tmp_dir) / mode / "fg"
    dir_wav.mkdir(parents=True, exist_ok=True)

    for key, value in tqdm(dict_fg.items()):
        s, sr = sf.read(value["original_path"])

        global_power = (s**2).mean()
        threshold = global_power * 10 ** (-clip_db / 10)

        # calc onset
        onset = 0
        for sec in range(len(s) // sr):
            local_power = (s[sec * sr : (sec + 1) * sr] ** 2).mean()
            if local_power > threshold:
                onset = sec
                break

        # calc offset
        offset = len(s) // sr
        for sec in range(len(s) // sr, 0, -1):
            local_power = (s[(sec - 1) * sr : sec * sr] ** 2).mean()
            if local_power > threshold:
                offset = sec
                break

        s = s[onset * sr : offset * sr]
        sf.write(dir_wav / f"{key}.wav", s, sr)

        value.update({"duration": len(s) / sr, "clip": str(dir_wav / f"{key}.wav")})
        dict_fg[key] = value

    with open(save_dir / "json" / f"fg_{mode}.json", "w") as f:
        json.dump(dict_fg, f, indent=4)


if __name__ == "__main__":
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    save_dir = Path(config["save_dir"])

    for mode in ["train", "valid", "test"]:
        preprocess(
            Path(config["save_dir"]),
            mode,
            Path(config["tmp_dir"]),
            config["clip_db"],
        )
