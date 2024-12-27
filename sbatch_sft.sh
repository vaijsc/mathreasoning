#!/bin/bash -e
#SBATCH --job-name=deepseek-math-sft-gsm8k-10epoch
#SBATCH --output=/home/hieupq1/hieupq1/math/logs/slurm_%x.out
#SBATCH --error=/home/hieupq1/hieupq1/math/logs/slurm_%x.err
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --mem-per-gpu=40GB
#SBATCH --cpus-per-gpu=16
#SBATCH --partition=applied
#SBATCH --mail-type=all
#SBATCH --mail-user=v.hieupq1@vinai.io
#SBATCH --dependency=114504 

module load python/miniconda3/miniconda3
save_dir="saves/deepseek-math-sft-gsm8k"
n_epoch=10
n_gpus=2

eval "$(conda shell.bash hook)"
conda activate llama_fac
cd /home/hieupq1/hieupq1/math/

bash LLaMA-Factory/sft.sh $save_dir $n_epoch $n_gpus
