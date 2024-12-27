#!/bin/bash -e
#SBATCH --job-name=deepseek-math-ul2-gsm8k-septoken-maskfull-sentence-equation-lossfulltarget-mixedcausalsenteqmasking-5ep-fullft
#SBATCH --output=/home/hieupq1/hieupq1/math/logs/slurm_%x.out
#SBATCH --error=/home/hieupq1/hieupq1/math/logs/slurm_%x.err
#SBATCH --nodes=2
#SBATCH --gpus-per-node=2
#SBATCH --mem-per-gpu=128GB
#SBATCH --cpus-per-gpu=32
#SBATCH --partition=applied
#SBATCH --mail-type=all
#SBATCH --mail-user=v.hieupq1@vinai.io
#SBATCH --dependency=116188
#SBATCH --exclude=sdc2-hpc-dgx-a100-001,sdc2-hpc-dgx-a100-002,sdc2-hpc-dgx-a100-009

JOB_NAME="deepseek-math-ul2-gsm8k-septoken-maskfull-sentence-equation-lossfulltarget-mixedcausalsenteqmasking-5ep-fullft"
save_dir="saves/${JOB_NAME}"
n_gpus=2
datasets="gsm8k_train_5_ul2_mixedcausalsenteqmasking"
ul2_causal=false
num_nodes=2

module load python/miniconda3/miniconda3
eval "$(conda shell.bash hook)"
conda activate llama_fac
cd /home/hieupq1/hieupq1/math/

bash LLaMA-Factory/ul2_full.sh $save_dir $n_gpus $datasets $ul2_causal $num_nodes
