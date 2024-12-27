#!/bin/bash -e

python LLaMA-Factory/src/utils/test_new_infer.py \
    --lora_path /home/hieupq1/hieupq1/math/saves/deepseek-math-ul2-gsm8k-sep-token-denoised-lossfulltarget/checkpoint-464 \
    --out_file infer_res/ul2-gsm8k-464-septoken-lossfulltarget.json \
    --batch_size 2 \
    --ul2 \
    --load_extra_param \
    --text_file ./test.txt \
    --ul2_sentinel_sdenoise
