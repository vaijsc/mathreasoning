#!/bin/bash -e

#SBATCH --job-name=mistral-7b-sft-gsm8k-10epoch-masked-thought
#SBATCH --output=/home/duongnt120/duongnt120/project/mathreasoning/logs/slurm_%x.out
#SBATCH --error=/home/duongnt120/duongnt120/project/mathreasoning/logs/slurm_%x.err
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-gpu=40GB
#SBATCH --cpus-per-gpu=16
#SBATCH --partition=research
#SBATCH --mail-type=all
#SBATCH --mail-user=v.hieupq1@vinai.io

module load python/miniconda3/miniconda3

model_path="Mistral-7B-v0.1"
save_dir="saves/mistral-7b-sft-gsm8k-10epoch-masked-thought"
datasets="gsm8k_train"
template="mistral"
n_epoch=10
n_gpus=1
masked_thought=0.4
num_new_tokens=1

eval "$(conda shell.bash hook)"
conda activate llama_fac
cd /home/duongnt120/duongnt120/project/mathreasoning/

bash LLaMA-Factory/sft.sh $model_path $save_dir $n_epoch $n_gpus $datasets $template $masked_thought $num_new_tokens
