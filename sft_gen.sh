#!/bin/bash -e
#SBATCH --job-name=sft-gen-deepseek-math-sft-gsm8k-348-sc-prod
#SBATCH --output=/home/hieupq1/hieupq1/math/logs/slurm_%x.out
#SBATCH --error=/home/hieupq1/hieupq1/math/logs/slurm_%x.err
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-gpu=40G
#SBATCH --cpus-per-gpu=16
#SBATCH --partition=applied
#SBATCH --mail-type=all
#SBATCH --mail-user=v.hieupq1@vinai.io
#SBATCH --dependency=114517


module load python/miniconda3/miniconda3
eval "$(conda shell.bash hook)"
conda activate llama_fac
cd /home/hieupq1/hieupq1/math/

python LLaMA-Factory/src/utils/infer_ul2.py \
    --lora_path /home/hieupq1/hieupq1/math/saves/deepseek-math-sft-gsm8k/checkpoint-348/ \
    --out_file infer_res/sft_gsm8k_348_base_sc_prod.json \
    --batch_size 2 \
    --sc "prod"
