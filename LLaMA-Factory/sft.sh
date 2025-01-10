#! /bin/bash
set -e

model_path=$1
output_dir=$2
n_epoch=$3
n_gpus=$4
datasets=$5
template=$6
masked_thought=$7
num_new_tokens=$8

if [ -z "$num_new_tokens" ]; then
  # default value for a dependency condition
  num_new_tokens=100 
fi

if [ -z "$masked_thought" ]; then
  # default value for a dependency condition
  masked_thought=-1 
fi

if [ -z "$template" ]; then
  # default value for a dependency condition
  template="deepseek-math"
fi

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

is_port_in_use() {
    lsof -i:$1 > /dev/null 2>&1
    return $?
}

# Loop until a free port is found within the specified range
while true; do
    random_port=$((RANDOM % 91 + 60010))  # Generate random port between 60010 and 60100
    
    if ! is_port_in_use $random_port; then
        echo "Selected free port: $random_port"
        break
    fi
done

deepspeed --include=$include \
    --master_port $random_port \
    LLaMA-Factory/src/train.py \
    --deepspeed LLaMA-Factory/examples/deepspeed/ds_z2_config.json \
    --model_name_or_path $model_path \
    --stage sft \
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
	  --warmup_ratio 0.05 \
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
    --gradient_accumulation_steps $(expr 64 / $n_gpus) \
    --masked_thought $masked_thought \
    --num_new_tokens $num_new_tokens
