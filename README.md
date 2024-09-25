# Lighthouse-Wrapper-for-Audio-Moment-Retrieval

## What is this?
This repository provides the procedure to conduct experiments with [Lighthouse](https://github.com/line/lighthouse) for the paper ["Language-based Audio Moment Retrieval" (under review)](https://arxiv.org/abs/2409.15672).
In addition, it supports the following functionalities (coming soon):
- Generation of Clotho-Moments from Clotho and UnAv-100
- Extraction of CLAP Features
- Evaluation of Zero-shot Sound Event Detection

## How to train/evaluate AMR models with Lighthouse?
1. Install [Lighthouse](https://github.com/line/lighthouse)

2. Download extracted CLAP features of Clotho-Moment/UnAV100-subset/TUT Sound Events 2017 from [here](https://zenodo.org/records/13806234)
    - You can also download wav files from [here](https://zenodo.org/records/13836117)

3. Set the path to the downloaded features in "(LIGHTHOUSE_PATH)/features".
    - For example, if you downloaded Clotho-Moment features, set the path to "(LIGHTHOUSE_PATH)/features/clotho-moment".

4. Run the following command to train the AMR model:
    ```bash
    python training/train.py --model qd_detr --dataset clotho-moment --feature clap
    ```

5. Run the following command to evaluate the AMR model:
    ```bash
    model=qd_detr
    dataset=unav100-subset
    feature=clap
    model_path={lighthouse_dir}/results/qd_detr/clotho-moment/clap/best.ckpt
    eval_split_name=val
    eval_path=data/unav100-subset/unav100-subset_test_release.jsonl

    python training/evaluate.py \
            --model $model \
            --dataset $dataset \
            --feature $feature \
            --model_path $model_path \
            --eval_split_name $eval_split_name \
            --eval_path $eval_path
    ```

## Generation of Clotho-Moments
Under construction

## Extraction of CLAP Features
Under construction

## Evaluation of Zero-shot Sound Event Detection
Under construction





