import argparse
from pathlib import Path

import numpy as np
import torch
from msclap import CLAP
from torch.nn import functional as F
from tqdm import tqdm


def dump_audio(data_dir, save_dir, extractor, model_name):
    list_data = sorted(data_dir.glob("*.wav"))
    print(list_data)
    if len(list_data) == 0:
        print(f"No audio files found in {data_dir}")
        return

    # Setup the directory to save the audio features"
    dir_save_feats = save_dir / f"{model_name}_text"
    dir_save_feats.mkdir(exist_ok=True, parents=True)

    # Loop through the audio files and extract the audio featseddings
    print("dump audio data...")
    for path_wav in tqdm(list_data):
        path_feats = dir_save_feats / f"{path_wav.stem}.npz"
        if path_feats.exists():
            continue
        feat, proj_feat = extractor.extract_audio_feats(str(path_wav))

        np.savez(path_feats, features=feat)


class ClapExtractor:
    def __init__(self, win_sec, hop_sec):
        # if gpu is available, use it
        self.use_cuda = torch.cuda.is_available()
        self.wrapper = CLAP(use_cuda=self.use_cuda, version="2023")
        if self.use_cuda:
            self.wrapper.clap.caption_encoder = self.wrapper.clap.caption_encoder.cuda()
            print("Inference on GPU")
        else:
            print("Inference on CPU")

        self.sl_win = SlidingWindos(win_sec, hop_sec)

    @torch.no_grad()
    def extract_audio_feats(self, path_wav):
        audio, sr = self.wrapper.read_audio(path_wav, resample=True)
        frames = self.sl_win(audio[0], sr)
        frames = frames.cuda() if self.use_cuda else frames

        feats = self.wrapper.clap.audio_encoder.base(frames)["embedding"]
        proj_feats = self.wrapper.clap.audio_encoder.projection(feats)

        feats = feats.cpu().numpy()
        proj_feats = proj_feats.cpu().numpy()

        return feats, proj_feats


class SlidingWindos:
    def __init__(self, win_sec, hop_sec):
        self.win_sec = win_sec
        self.hop_sec = hop_sec

    def __call__(self, audio, sr):
        """
        Perform sliding window processing on a 1D tensor with center-based cutting.

        Parameters:
        audio (torch.tensor): 1D tensor.
        win_sec (float): Length of each window.
        hop_sec (float): Number of elements to move the window at each step.
        sr (int): Sampling rate.

        Returns:
        torch.tensor: 2D tensor with shape (num_windows, win_length).
        """
        if audio.ndim != 1:
            raise ValueError("Input audio must be 1D tensor.")

        win_length = int(self.win_sec * sr)
        hop_length = int(self.hop_sec * sr)

        half_win = win_length // 2
        padded_audio = F.pad(audio, (half_win, half_win), mode="constant", value=0)
        windows = padded_audio.unfold(0, win_length, hop_length)

        return torch.tensor(windows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=Path, help="audio data directory")
    parser.add_argument("save_dir", type=Path, help="directory to save the features")
    parser.add_argument("--win_sec", type=float, default=1.0, help="window length")
    parser.add_argument("--hop_sec", type=float, default=1.0, help="hop length")
    parser.add_argument("--model_name", default="clap", help="model name")
    args = parser.parse_args()

    if args.model_name == "clap":
        extractor = ClapExtractor(args.win_sec, args.hop_sec)
    else:
        raise ValueError(f"Invalid model name: {args.model_name}")

    dump_audio(
        args.data_dir,
        args.save_dir,
        extractor,
        args.model_name,
    )
