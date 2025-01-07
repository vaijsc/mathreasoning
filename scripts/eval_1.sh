#!/bin/bash -e
#SBATCH --job-name=gen_deepseek-7b-ul2-gsm8k-t5-rerun-correctdata-finetune-lmhead-1e-4_checkpoint-928_beamsearch_ul2_1
#SBATCH --output=/home/duongnt120/duongnt120/project/mathreasoning/logs/slurm_%x.out
#SBATCH --error=/home/duongnt120/duongnt120/project/mathreasoning/logs/slurm_%x.err
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
cd /home/duongnt120/duongnt120/project/mathreasoning/

python LLaMA-Factory/src/utils/infer_ul2.py \
    --base_path deepseek-math-7b-base \
    --lora_path saves/deepseek-7b-ul2-gsm8k-t5-rerun-correctdata-finetune-lmhead-1e-4/checkpoint-928 \
    --dataset_path cache/gsm8k_test_1.json \
    --out_file infer_res/deepseek-7b-ul2-gsm8k-t5-rerun-correctdata-finetune-lmhead-1e-4_checkpoint-928_gsm8k_test_beamsearch_ul2_1.json \
    --batch_size 1 \
    --ul2 \
    --sc none
