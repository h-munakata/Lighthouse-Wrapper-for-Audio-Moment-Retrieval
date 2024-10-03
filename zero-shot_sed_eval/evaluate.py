import argparse
import json

import numpy as np
from sklearn.metrics import precision_recall_fscore_support


def load_ref_as_array(ref_jsonl, label_resolution):
    with open(ref_jsonl, "r") as f:
        lines = f.readlines()
    data = [json.loads(line) for line in lines]

    # get label set
    labels = []
    for item in data:
        query = item["query"]
        if query not in labels:
            labels.append(query)

    er_dict = {}
    for num, item in enumerate(data):
        # load data
        vid = item["vid"]
        query = item["query"]
        duration = item["duration"]
        relevant_windows = item["relevant_windows"]

        label_idx = labels.index(query)
        total_frames = int(np.ceil(duration * label_resolution))
        time_axis = np.arange(total_frames)

        # load nparray
        er = er_dict.get(vid, np.zeros((len(labels), total_frames)))

        # load start and end time as nparray
        for t in relevant_windows:
            ts, te = t
            active_segment = (time_axis >= ts) * (time_axis <= te)
            er[label_idx, :] += active_segment

        er_dict[vid] = er

    # binarize
    for k, v in er_dict.items():
        er_dict[k] = er_dict[k] > 0

    return er_dict, labels


def load_pred_as_array(pred_jsonl, ref_er_dict, labels, threshold, label_resolution):
    with open(pred_jsonl, "r") as f:
        lines = f.readlines()
    data = [json.loads(line) for line in lines]

    er_dict = {}
    for num, item in enumerate(data):
        # load data
        vid = item["vid"]
        query = item["query"]
        relevant_windows = item["pred_relevant_windows"]

        # load nparray
        duration = ref_er_dict[vid].shape[1]
        er = er_dict.get(vid, np.zeros((len(labels), duration)))

        label_idx = labels.index(query)
        total_frames = int(np.ceil(duration / label_resolution))
        time_axis = np.arange(total_frames * label_resolution)

        # load start and end time as nparray
        for t in relevant_windows:
            ts, te, score = t
            active_segment = (time_axis >= ts) * (time_axis <= te)
            if score > threshold:
                er[label_idx, :] += active_segment

        er_dict[vid] = er

    # binarize
    for k, v in er_dict.items():
        er_dict[k] = er_dict[k] > 0

    return er_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ref_jsonl", type=str, help="target jsonl file")
    parser.add_argument("pred_jsonl", type=str, help="prediction jsonl file")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--label_resolution", type=float, default=1)
    args = parser.parse_args()

    ref_er_dict, labels = load_ref_as_array(args.ref_jsonl, args.label_resolution)
    pred_er_dict = load_pred_as_array(
        args.pred_jsonl,
        ref_er_dict,
        labels,
        args.threshold,
        args.label_resolution,
    )

    # evaluate
    ref_frames, pred_frames = [], []
    for vid in ref_er_dict.keys():
        ref_frames.append(ref_er_dict[vid])
        pred_frames.append(pred_er_dict[vid])

    ref_frames = np.hstack(ref_frames)
    pred_frames = np.hstack(pred_frames)

    precision, recall, fscore, _ = precision_recall_fscore_support(
        ref_frames, pred_frames, average="micro", zero_division=0
    )
    print("Micro Average")
    print("Precision, Recall, F1")
    print(f"{precision * 100:.2f}, {recall * 100:.2f}, {fscore * 100:.2f}")
