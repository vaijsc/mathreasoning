#!/bin/bash -e
#saves/deepseek-math-test-dpo-580/checkpoint-1758/
#err_analysis/err-deepseek-math-ul2-gsm8k-septoken-maskfull-sentence-equation-lossfulltarget-5ep-bartmixed-singlemasktoken_checkpoint-580_gsm8k_test.json
python LLaMA-Factory/src/utils/infer_ul2.py \
    --base_path deepseek-math-7b-base/ \
    --lora_path saves/deepseek-math-ul2-gsm8k-forward-backward/checkpoint-1387 \
    --dataset_path LLaMA-Factory/data/gsm8k_sample.json \
    --out_file infer_res/deepseek-math-ul2-gsm8k-forward-backward-test_save.json \
    --batch_size 1 \
    --sc cot \
    --ul2 \
    --bart_mixed \
    --cot_mode "greedy" \
    --cot_threshold 0.6 \
    --ul2_rethinking mixed \
