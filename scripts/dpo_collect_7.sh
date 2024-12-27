#!/bin/bash -e
#SBATCH --job-name=gen_deepseek-math-ul2-gsm8k-septoken-maskfull-sentence-equation-lossfulltarget-5ep-bartmixed-singlemasktoken_checkpoint-580_dpo_cot_rethink_7
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
    --lora_path /home/hieupq1/hieupq1/math/saves/deepseek-math-ul2-gsm8k-septoken-maskfull-sentence-equation-lossfulltarget-5ep-bartmixed-singlemasktoken/checkpoint-580 \
    --dataset_path cache/gsm8k_train_7.json \
    --out_file infer_res/dpo_collect_cot/deepseek-math-ul2-gsm8k-septoken-maskfull-sentence-equation-lossfulltarget-5ep-bartmixed-singlemasktoken_checkpoint-580_gsm8k_train_dpo_cot_rethink_bartmixed0616_7.json \
    --batch_size 1 \
    --ul2 \
    --sc cot \
    --cot_mode "greedy" \
    --bart_mixed \
    --ul2_rethinking mixed \
    --cot_threshold 0.6 \
    --collect_wrong_chain
