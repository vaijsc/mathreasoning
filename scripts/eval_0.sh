#!/bin/bash -e
#SBATCH --job-name=gen_mistral-7b-ul2-gsm8k-t5-rerun-correctdata-finetune-lmhead_checkpoint-928_beamsearch_ul2_0
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
    --base_path Mistral-7B-v0.1 \
    --lora_path /home/hieupq1/hieupq1/math/saves/mistral-7b-ul2-gsm8k-t5-rerun-correctdata-finetune-lmhead/checkpoint-928 \
    --dataset_path cache/gsm8k_test_0.json \
    --out_file infer_res/mistral-7b-ul2-gsm8k-t5-rerun-correctdata-finetune-lmhead_checkpoint-928_gsm8k_test_beamsearch_ul2_0.json \
    --batch_size 1 \
    --ul2 \
    --sc none
