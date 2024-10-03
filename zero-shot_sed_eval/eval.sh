#!/bin/bash

ref_jsonl=/LIGHTHOUSE/PATH/data/tut2017/tut2017_test_release.jsonl
pred_jsonl=/YOUR/PREDICTED/RESULT/IN/LIGHTHOUSE/hl_val_submission.jsonl

python evaluate.py $ref_jsonl $pred_jsonl
