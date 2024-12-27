#! /bin/bash
set -e

model_path="deepseek-math-7b-base"
template="deepseek-math"
output_dir=$1
n_epoch=1
n_gpus=1
n_ul2_epoch=6
# cosine_scheduler_epoch=$6

# Initialize variables
python LLaMA-Factory/src/train.py \
	--model_name_or_path $model_path \
	--stage ul2 \
	--dataset gsm8k_train \
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
	--preprocessing_num_workers 16 \
	--lr_scheduler_type cosine \
	--logging_steps 10 \
	--warmup_steps 35 \
	--save_steps 100000 \
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
	--gradient_accumulation_steps $(expr 64 / $n_gpus)  \
	--ul2_nepochs $n_ul2_epoch \
  --ul2_finetune_embedding true \
  --use_unsloth true
# --cosine_scheduler_epoch $cosine_scheduler_epoch \
