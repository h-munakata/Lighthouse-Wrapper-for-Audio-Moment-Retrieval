import argparse
import json
from pathlib import Path

import numpy as np
import torch
from msclap import CLAP
from tqdm import tqdm


def save_text(data_path, save_dir, extractor, model_name):
    if not data_path.exists():
        print(f"{data_path} does not exist.")
        return

    with open(data_path) as f:
        data = [json.loads(line) for line in f]

    # Setup the directory to save the text features"
    dir_save_feats = save_dir / f"{model_name}_text"
    dir_save_feats.mkdir(exist_ok=True, parents=True)

    print("save text data...")
    for _d in tqdm(data):
        feat, proj_feat = extractor.extract_text_feats(_d["query"])

        path_feats = dir_save_feats / f"qid{_d['qid']}.npz"
        np.savez(path_feats, last_hidden_state=feat)


class ClapExtractor:
    def __init__(self):
        # if gpu is available, use it
        self.use_cuda = torch.cuda.is_available()
        self.wrapper = CLAP(use_cuda=self.use_cuda, version="2023")
        self.text_enc = self.wrapper.clap.caption_encoder
        if self.use_cuda:
            print("Inference on GPU")
            self.text_enc = self.text_enc.cuda()
        else:
            print("Inference on CPU")

    @torch.no_grad()
    def extract_text_feats(self, text):
        x = self.wrapper.preprocess_text([text])
        mask = x["attention_mask"]
        len_output = torch.sum(mask, dim=-1, keepdims=True)
        out = self.text_enc.base(**x)
        hidden_states = out[0]
        pooled_output = out[1]

        if "clip" in self.text_enc.text_model:
            out = self.clip_text_projection(pooled_output)  # get CLS token output
        elif "gpt" in self.text_enc.text_model:
            batch_size = x["input_ids"].shape[0]
            sequence_lengths = (
                torch.ne(x["input_ids"], 0).sum(-1) - 1
            )  # tensor([13, 14, 18, 17])
            out = hidden_states[
                torch.arange(batch_size, device=hidden_states.device),
                sequence_lengths,
            ]  # [batch_size, 768] = [4, 768]
        else:
            out = hidden_states[:, 0, :]  # get CLS token output

        projected_feat = self.text_enc.projection(out)

        feat = hidden_states[0, :len_output].cpu().numpy()
        proj_feat = projected_feat.cpu().numpy()

        return feat, proj_feat


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=Path, default=None, help="text data path")
    parser.add_argument("save_dir", type=Path, help="directory to save the features")
    parser.add_argument("--model_name", default="clap", help="model name")
    args = parser.parse_args()

    if args.model_name == "clap":
        extractor = ClapExtractor()
    else:
        raise ValueError(f"Invalid model name: {args.model_name}")

    save_text(
        args.data_dir,
        args.save_dir,
        extractor,
        args.model_name,
    )
