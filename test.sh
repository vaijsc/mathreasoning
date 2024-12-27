#!/bin/bash

python LLaMA-Factory/src/utils/infer_ul2.py \
    --lora_path /home/hieupq1/hieupq1/math/saves/deepseek-math-ul2-gsm8k-septoken-maskfull-sentence-equation-lossfulltarget-5ep-bartmixed-singlemasktoken/checkpoint-812/ \
    --dataset_path cache/gsm8k_test_0.json \
    --out_file infer_res/deepseek-math-ul2-gsm8k-septoken-maskfull-sentence-equation-lossfulltarget-5ep-bartmixed-singlemasktoken_checkpoint-812_gsm8k_test_0.json \
    --batch_size 1 \
    --ul2 \
    --beam_size 5 \
    --ul2_rethinking mixed
