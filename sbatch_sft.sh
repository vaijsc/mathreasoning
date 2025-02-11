#!/bin/bash -e
#SBATCH --job-name=llama3-sft-gsm8k-10epoch-re-lmheadft
#SBATCH --output=/home/duongnt120/duongnt120/mathreasoning/logs/slurm_%x.out
#SBATCH --error=/home/duongnt120/duongnt120/mathreasoning/logs/slurm_%x.err
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-gpu=40GB
#SBATCH --cpus-per-gpu=16
#SBATCH --partition=research
#SBATCH --mail-type=all
#SBATCH --mail-user=v.hieupq1@vinai.io

module load python/miniconda3/miniconda3

model_path="Llama-3.1-8B"
save_dir="saves/llama3-sft-gsm8k-10epoch-re-lmheadft"
datasets="gsm8k_train"
template="llama3-math"
n_epoch=10
n_gpus=1

eval "$(conda shell.bash hook)"
conda activate llama_fac
cd /home/duongnt120/duongnt120/mathreasoning/

bash LLaMA-Factory/sft.sh $model_path $save_dir $n_epoch $n_gpus $datasets $template
