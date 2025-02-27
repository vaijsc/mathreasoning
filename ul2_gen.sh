#!/bin/bash -e

#SBATCH --job-name=ul2-gen-1276-deepseek-math-ul2-gsm8k-septoken-maskfull-sentence-equation-lossfulltarget-2-mixedcausalsenteqmasking-5ep
#SBATCH --output=/home/hieupq1/hieupq1/math/logs/slurm_%x.out
#SBATCH --error=/home/hieupq1/hieupq1/math/logs/slurm_%x.err
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-gpu=40G
#SBATCH --cpus-per-gpu=16
#SBATCH --partition=research
#SBATCH --mail-type=all
#SBATCH --mail-user=v.hieupq1@vinai.io
#SBATCH --dependency=120925
#SBATCH --exclude=sdc2-hpc-dgx-a100-002,sdc2-hpc-dgx-a100-001,sdc2-hpc-dgx-a100-010,sdc2-hpc-dgx-a100-009

# --lora_path saves/deepseek-math-ul2-gsm8k-septoken-maskfull-sentence-equation-lossfulltarget-mixedcausalsenteqmasking-5ep/checkpoint-812 \
# --base_path Mistral-7B-v0.1/ \
#saves/mistral-7b-ul2-gsm8k-t5-rerun-correctdata-finetune-lmhead/checkpoint-812/
# --lora_path saves/mistral-7b-ul2-gsm8k-t5-rerun-correctdata/checkpoint-812/ \
module load python/miniconda3/miniconda3
eval "$(conda shell.bash hook)"
conda activate llama_fac

python LLaMA-Factory/src/utils/infer_ul2.py \
    --lora_path saves/llama31-sft-gsm8k-masked-thought/checkpoint-1160/ \
    --base_path Llama-3.1-8B \
    --dataset LLaMA-Factory/data/gsm8k_test.json \
    --out_file infer_res/test_qwen.json \
    --batch_size 1 \
    --num_new_token 1 \
    --masked_thought \
    --causal_prefix \
    --sc greedy
