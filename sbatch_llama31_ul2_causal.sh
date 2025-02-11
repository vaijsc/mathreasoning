#!/bin/bash -e

#SBATCH --job-name=llama31-ul-gsm-llama-math-template-causalpref
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

JOB_NAME="llama31-ul-gsm-llama-math-template-causalpref"                                                                                                
save_dir="saves/${JOB_NAME}"                                                                          
n_gpus=2                                  
model_path="Llama-3.1-8B"                                                                                          
datasets="gsm8k_train_5_ul2_mixedcausalsenteqmasking"                                                                        
ul2_causal=true                                                                                                                 
template="llama3-math"                                  
                                                                                                                     
module load python/miniconda3/miniconda3
eval "$(conda shell.bash hook)"
conda activate llama_fac
cd /home/duongnt120/duongnt120/mathreasoning/

bash LLaMA-Factory/ul2.sh $model_path $save_dir $n_gpus $datasets $ul2_causal $template   
