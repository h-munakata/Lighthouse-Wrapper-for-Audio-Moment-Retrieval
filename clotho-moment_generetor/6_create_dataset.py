import json
import os
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import yaml
from tqdm import tqdm


def _gen_audio(data, wav_dir, duration, sr, bg_db=None):
    path_bg = data["bg"]["path"]
    if bg_db is None:
        bg_db = data["bg"]["dB"]
    data_name = f"{Path(path_bg).parent.stem}_{Path(path_bg).stem}"
    save_path = wav_dir / f"{data_name}.wav"
    if os.path.exists(save_path):
        return
    s_bg, _ = librosa.load(path_bg, sr=sr)

    power_bg = np.mean(s_bg**2)
    s_bg = 10 ** (bg_db / 20) * s_bg

    for fg in data["fg"]:
        # add fg to bg
        s_fg, _ = librosa.load(fg["path"], sr=sr)
        s_fg = s_fg / np.max(np.abs(s_fg))

        weight = 10 ** (fg["dB"] / 20) * np.sqrt(power_bg)
        s_fg = weight * s_fg

        start_time = fg["start_time"]
        start_sample = int(start_time * sr)

        s_bg[start_sample : start_sample + len(s_fg)] += s_fg

    # save wav
    s_bg = 1 / np.max(np.abs(s_bg)) * s_bg
    sf.write(str(save_path), s_bg, sr)


def _gen_text(data, wav_dir, duration, bg_db=None):
    path_bg = data["bg"]["path"]
    data_name = f"{Path(path_bg).parent.stem}_{Path(path_bg).stem}"
    list_info = []

    for fg in data["fg"]:
        qid = fg["qid"]
        start = fg["start_time"]
        end = start + fg["duration"]
        caption = fg["caption"]

        _info = {
            "qid": f"{qid:05d}",
            "query": caption,
            "duration": duration,
            "vid": data_name,
            "relevant_windows": [[float(f"{start:.1f}"), float(f"{end:.1f}")]],
            "fg_dB": fg["dB"],
        }
        if bg_db is not None:
            _info["bg_dB"] = bg_db
        list_info.append(_info)

    return list_info


def generate_data(save_dir, mode, duration, sr, bg_db):
    path_recipe = Path(config["save_dir"]) / "json" / f"recipe_{mode}.json"
    wav_dir = Path(config["save_dir"]) / "wav" / f"{sr}hz" / mode
    if bg_db is not None:
        wav_dir = wav_dir / f"{bg_db}dB"
    text_dir = Path(config["save_dir"]) / "text"
    os.makedirs(wav_dir, exist_ok=True)
    os.makedirs(text_dir, exist_ok=True)
    with open(path_recipe) as f:
        recipe = json.load(f)

    # Generate text data
    print(f"Generate {mode} text data")
    if os.path.exists(text_dir / f"{mode}.jsonl"):
        print(f"{mode}.jsonl already exists")
    else:
        for data in tqdm(recipe):
            list_info = _gen_text(data, wav_dir, duration, bg_db)
            for _info in list_info:
                # Save the query as jsonl
                with open(text_dir / f"{mode}.jsonl", "a") as f:
                    json.dump(_info, f)
                    f.write("\n")

    # Generate audio data
    print(f"Generate {mode} audio data")
    map_fn = partial(_gen_audio, wav_dir=wav_dir, duration=duration, bg_db=bg_db, sr=sr)
    with Pool(processes=16) as p:
        list(tqdm(p.imap_unordered(map_fn, recipe), total=len(recipe)))


if __name__ == "__main__":
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    for mode in ["train", "valid", "test"]:
        generate_data(
            config["save_dir"], mode, config["clip_duration"], config["sr"], bg_db=None
        )
