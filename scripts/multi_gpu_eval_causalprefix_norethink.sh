#!/bin/bash -ex

num_gpu=$1
dataset_path=$2
output_dataset_prefix=$3
model_name=$4
checkpoint_name=$5
ul2_rethink=$6
model_path="saves/$model_name/$checkpoint_name" # deepseek-math-ul2-gsm8k-septoken-maskfull-sentence-equation-lossfulltarget-mixedcausalsenteqmasking-5ep/checkpoint-696/
dependency=$7

if [ -z "$dependency" ]; then
  # default value for a dependency condition
  dependency=1 
fi

if [ -z "$ul2_rethink" ]; then
  # default value for a dependency condition
  ul2_rethink="mixed" 
fi

job_script_content='#!/bin/bash -e
#SBATCH --job-name=gen_{{model_name}}_{{checkpoint_name}}_norethink_bartmixed_{{cur_partition}}
#SBATCH --output=/home/hieupq1/hieupq1/math/logs/slurm_%x.out
#SBATCH --error=/home/hieupq1/hieupq1/math/logs/slurm_%x.err
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-gpu=40G
#SBATCH --cpus-per-gpu=16
#SBATCH --partition=research
#SBATCH --mail-type=all
#SBATCH --mail-user=v.hieupq1@vinai.io
#SBATCH --dependency={{dependency}}
#SBATCH --exclude=sdc2-hpc-dgx-a100-002,sdc2-hpc-dgx-a100-001,sdc2-hpc-dgx-a100-010,sdc2-hpc-dgx-a100-009

module load python/miniconda3/miniconda3
eval "$(conda shell.bash hook)"
conda activate llama_fac
cd /home/hieupq1/hieupq1/math/

python LLaMA-Factory/src/utils/infer_ul2.py \
    --lora_path /home/hieupq1/hieupq1/math/saves/{{model_name}}/{{checkpoint_name}} \
    --dataset_path cache/{{dataset_split}} \
    --out_file infer_res/{{model_name}}_{{checkpoint_name}}_{{output_dataset_prefix}}_norethink_{{cur_partition}}.json \
    --batch_size 1 \
    --ul2 \
    --sc cot \
    --cot_mode "greedy" \
    --bart_mixed \
    --causal_prefix
'

bash scripts/split_file.sh $dataset_path $num_gpu $output_dataset_prefix

for ((cur_partition=0; cur_partition<num_gpu; cur_partition++)); do
  dataset_split="${output_dataset_prefix}_${cur_partition}.json" 
  job_script_filled=$(echo "$job_script_content" \
                      | sed "s/{{model_name}}/${model_name}/g" \
                      | sed "s/{{checkpoint_name}}/${checkpoint_name}/g" \
                      | sed "s/{{output_dataset_prefix}}/${output_dataset_prefix}/g" \
                      | sed "s/{{cur_partition}}/${cur_partition}/g" \
                      | sed "s/{{dependency}}/${dependency}/g" \
                      | sed "s/{{dataset_split}}/${dataset_split}/g") 

  echo "$job_script_filled" > "scripts/eval_${cur_partition}.sh"
  sbatch scripts/eval_${cur_partition}.sh
done
