#!/bin/bash -e
#SBATCH --job-name=gen_deepseek-math-sft-gsm8k-masked-thought_checkpoint-928_2_beamsearch_norethink
#SBATCH --output=/home/hieupq1/hieupq1/math/logs/slurm_%x.out
#SBATCH --error=/home/hieupq1/hieupq1/math/logs/slurm_%x.err
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-gpu=40G
#SBATCH --cpus-per-gpu=16
#SBATCH --partition=research
#SBATCH --mail-type=all
#SBATCH --mail-user=v.hieupq1@vinai.io
#SBATCH --dependency=1
#SBATCH --exclude=sdc2-hpc-dgx-a100-002,sdc2-hpc-dgx-a100-001,sdc2-hpc-dgx-a100-010,sdc2-hpc-dgx-a100-009

module load python/miniconda3/miniconda3
eval "$(conda shell.bash hook)"
conda activate llama_fac
cd /home/hieupq1/hieupq1/math/

python LLaMA-Factory/src/utils/infer_ul2.py \
    --lora_path /home/hieupq1/hieupq1/math/saves/deepseek-math-sft-gsm8k-masked-thought/checkpoint-928 \
    --dataset_path cache/gsm8k_test_2.json \
    --out_file infer_res/deepseek-math-sft-gsm8k-masked-thought_checkpoint-928_gsm8k_test_beamsearch_2.json \
    --batch_size 1 \
    --causal_prefix \
    --sc none
