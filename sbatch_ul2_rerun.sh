#!/bin/bash -e

#SBATCH --job-name=deepseek-7b-ul2-gsm8k-t5-rerun
#SBATCH --output=/home/hieupq1/hieupq1/math/logs/slurm_%x.out
#SBATCH --error=/home/hieupq1/hieupq1/math/logs/slurm_%x.err
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --mem-per-gpu=40GB
#SBATCH --cpus-per-gpu=16
#SBATCH --partition=research
#SBATCH --mail-type=all
#SBATCH --mail-user=v.hieupq1@vinai.io
#SBATCH --dependency=116188
#SBATCH --exclude=sdc2-hpc-dgx-a100-001,sdc2-hpc-dgx-a100-002

JOB_NAME="deepseek-7b-ul2-gsm8k-t5-rerun"
save_dir="saves/${JOB_NAME}"
n_gpus=2
model_path="deepseek-math-7b-base"
# datasets="gsm8k_train_5_ul2_1_bartmixed"
datasets="gsm8k_train_5_ul2_mixedcausalsenteqmasking"
ul2_causal=false
template="deepseek-math"

#,gsm8k_train_0_ul2_1_bartmixedreverse

module load python/miniconda3/miniconda3
eval "$(conda shell.bash hook)"
conda activate llama_fac
cd /home/hieupq1/hieupq1/math/

bash LLaMA-Factory/ul2.sh $model_path $save_dir $n_gpus $datasets $ul2_causal $template
