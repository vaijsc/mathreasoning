#! /bin/bash
set -e

model_path=$1
output_dir=$2
n_epoch=1
n_gpus=$3
datasets=$4
ul2_causal=$5
template=$6

if [ -z "$template" ]; then
  # default value for a dependency condition
  template="deepseek-math" 
fi

# cosine_scheduler_epoch=$6

# Initialize variables
cuda_visible_devices=""
include=""

# Loop to construct CUDA_VISIBLE_DEVICES and --include parameters
for (( i=0; i<${n_gpus}; i++ )); do
  if [ $i -eq 0 ]; then
	cuda_visible_devices="$i"
	include="localhost:$i"
  else
	cuda_visible_devices="$cuda_visible_devices,$i"
	include="$include,$i"
  fi
done

deepspeed --include=$include \
  --master_port 60000 \
	LLaMA-Factory/src/train.py \
	--deepspeed LLaMA-Factory/examples/deepspeed/ds_z2_config.json \
	--model_name_or_path $model_path \
	--stage ul2 \
	--dataset $datasets \
	--do_train \
	--dataset_dir LLaMA-Factory/data \
	--template $template \
	--finetuning_type lora \
	--lora_target "all" \
	--lora_rank 32 \
	--output_dir $output_dir \
	--overwrite_cache \
	--overwrite_output_dir \
	--cutoff_len 4096 \
	--preprocessing_num_workers 32 \
	--lr_scheduler_type cosine \
	--logging_steps 10 \
	--warmup_ratio 0.05 \
	--eval_strategy "no" \
	--do_eval false \
	--save_strategy steps \
	--save_steps 116 \
	--learning_rate 5e-5 \
	--num_train_epochs $n_epoch \
	--val_size 0.0 \
	--ddp_timeout 180000000 \
	--bf16 \
	--per_device_train_batch_size 1 \
	--per_device_eval_batch_size 1 \
	--gradient_accumulation_steps $(expr 64 / $n_gpus) \
  --ul2_finetune_embedding true \
  --ul2_causal $ul2_causal
# --cosine_scheduler_epoch $cosine_scheduler_epoch \
