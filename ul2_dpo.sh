#! /bin/bash -e

#SBATCH --job-name=test_dp_beta001-bs256-dpop-10ep-1e-7
#SBATCH --output=/home/hieupq1/hieupq1/math/logs/slurm_%x.out
#SBATCH --error=/home/hieupq1/hieupq1/math/logs/slurm_%x.err
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --mem-per-gpu=40GB
#SBATCH --cpus-per-gpu=16
#SBATCH --partition=research
#SBATCH --mail-type=all
#SBATCH --mail-user=v.hieupq1@vinai.io
#SBATCH --dependency=116188
#SBATCH --exclude=sdc2-hpc-dgx-a100-001,sdc2-hpc-dgx-a100-002

module load python/miniconda3/miniconda3
eval "$(conda shell.bash hook)"
conda activate llama_fac
cd /home/hieupq1/hieupq1/math/

JOB_NAME="deepseek-math-test-dpo-580-reflora-beta001-bs256-dpop-10ep-1e-7"
output_dir="saves/${JOB_NAME}"
n_gpus=4
model_path="deepseek-math-7b-base"
datasets="test_dpo"
lora_path="saves/deepseek-math-ul2-gsm8k-septoken-maskfull-sentence-equation-lossfulltarget-5ep-bartmixed-singlemasktoken/checkpoint-580/"
causal_prefix=false
template="deepseek-math"

deepspeed --include="localhost:0,1,2,3" \
  --master_port 60000 \
	LLaMA-Factory/src/train.py \
	--deepspeed LLaMA-Factory/examples/deepspeed/ds_z2_config.json \
  --model_name_or_path $model_path \
	--adapter_name_or_path $lora_path \
  --ignore_data_skip \
	--stage ul2_dpo \
	--dataset $datasets \
	--do_train \
	--dataset_dir LLaMA-Factory/data \
	--template $template \
	--finetuning_type lora \
	--lora_target "all" \
	--lora_rank 32 \
  --pref_beta 0.01 \
  --ul2_finetune_embedding true \
	--output_dir $output_dir \
	--overwrite_cache \
	--overwrite_output_dir \
	--cutoff_len 4096 \
	--preprocessing_num_workers 32 \
	--lr_scheduler_type constant \
	--logging_steps 10 \
	--warmup_ratio 0.00 \
	--save_steps 100000 \
	--eval_strategy "no" \
	--do_eval false \
	--save_strategy steps \
	--save_steps 60 \
	--learning_rate 1e-7 \
	--num_train_epochs 10 \
	--val_size 0.0 \
	--ddp_timeout 180000000 \
	--bf16 \
	--per_device_train_batch_size 1 \
	--per_device_eval_batch_size 1 \
	--gradient_accumulation_steps $(expr 256 / $n_gpus) \
