#! /bin/bash
set -e

model_path="deepseek-math-7b-base"
template="deepseek-math"
output_dir=$1
n_epoch=1
n_gpus=$2
datasets=$3
ul2_causal=$4
num_nodes=$5

# Initialize variables
cuda_visible_devices=""
include=""

master_addr=$(getent hosts $(scontrol show hostname "$SLURM_JOB_NODELIST" | head -n 1) | awk '{ print $1 }')

hostfile="/home/hieupq1/hieupq1/math/cache/hosts.txt"
> $hostfile  # Clear the file if it exists
for node in $(scontrol show hostname $SLURM_JOB_NODELIST); do
    echo "$node slots=2" >> $hostfile
done

export NCCL_SOCKET_IFNAME=bond0
export FI_EFA_USE_DEVICE_RDMA=1
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=WARN
export CUDA_LAUNCH_BLOCKING=0

srun --jobid $SLURM_JOBID deepspeed -H $hostfile --no_ssh --node_rank $SLURM_NODEID --master_addr $master_addr --master_port 60000 --num_gpus=$n_gpus --num_nodes $num_nodes\
	LLaMA-Factory/src/train.py \
	--deepspeed LLaMA-Factory/examples/deepspeed/ds_z2_config.json \
	--model_name_or_path $model_path \
	--stage ul2 \
	--dataset $datasets \
	--do_train \
	--dataset_dir LLaMA-Factory/data \
	--template $template \
	--finetuning_type full \
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
	--gradient_accumulation_steps $(expr 64 / $n_gpus)  \
  --ul2_finetune_embedding true \
  --ul2_causal $ul2_causal
