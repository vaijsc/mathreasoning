#!/bin/bash -e

#SBATCH --job-name=mistral-ul-math-re
#SBATCH --output=/home/duongnt120/duongnt120/mathreasoning/logs/slurm_%x.out
#SBATCH --error=/home/duongnt120/duongnt120/mathreasoning/logs/slurm_%x.err
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --mem-per-gpu=40GB
#SBATCH --cpus-per-gpu=16
#SBATCH --partition=research
#SBATCH --mail-type=all
#SBATCH --mail-user=v.duongnt120@vinai.io
#SBATCH --dependency=116188
#SBATCH --exclude=sdc2-hpc-dgx-a100-015

JOB_NAME="mistral-ul-math-re"                                                                                                
save_dir="saves/${JOB_NAME}"                                                                          
n_gpus=2                                  
model_path="Mistral-7B-v0.1"                                                                                          
datasets="MATH_train_6_ul2_1"                                                                        
ul2_causal=false                                                                                                                 
template="mistral"                                  
                                                                                                                     
module load python/miniconda3/miniconda3
eval "$(conda shell.bash hook)"
conda activate llama_fac
cd /home/duongnt120/duongnt120/mathreasoning/

bash LLaMA-Factory/ul2.sh $model_path $save_dir $n_gpus $datasets $ul2_causal $template   
