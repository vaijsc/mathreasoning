#!/bin/bash -e

#SBATCH --job-name=deepseek-7b-ul2-random-8ep
#SBATCH --output=/home/duongnt120/duongnt120/project/mathreasoning/logs/slurm_%x.out
#SBATCH --error=/home/duongnt120/duongnt120/project/mathreasoning/logs/slurm_%x.err
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-gpu=40GB
#SBATCH --cpus-per-gpu=16
#SBATCH --partition=research
#SBATCH --mail-type=all
#SBATCH --mail-user=v.duongnt120@vinai.io
#SBATCH --dependency=116188
#SBATCH --exclude=sdc2-hpc-dgx-a100-001,sdc2-hpc-dgx-a100-002

JOB_NAME="deepseek-7b-ul2-random-8ep"
save_dir="saves/${JOB_NAME}"
n_gpus=1
model_path="deepseek-math-7b-base"
datasets="gsm8k_train_4_ul2_deepseek_random_mask"
ul2_causal=false
template="deepseek-math"
lr=5e-5
num_new_tokens=100

#,gsm8k_train_0_ul2_1_bartmixedreverse

module load python/miniconda3/miniconda3
eval "$(conda shell.bash hook)"
conda activate llama_fac
cd /home/duongnt120/duongnt120/project/mathreasoning/

bash LLaMA-Factory/ul2.sh $model_path $save_dir $n_gpus $datasets $ul2_causal $template $lr $num_new_tokens
