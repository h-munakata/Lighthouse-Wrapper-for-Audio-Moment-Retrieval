# !/bin/bash

win_sec=1
hop_sec=1
model_name=clap

data_dir=../clotho-moment_generetor/clotho-moment

for mode in train valid test; do
    python extract_text_feat.py \
    ${data_dir}/text/${mode}.jsonl \
    ${data_dir}/feature \
    --model_name=${model_name}
done

for mode in train valid test; do
    python extract_audio_feat.py \
        ${data_dir}/wav/${mode} \
        ${data_dir}/feature \
        --win_sec ${win_sec} \
        --hop_sec ${hop_sec} \
    --model_name=${model_name}
done
