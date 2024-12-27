#!/bin/bash -e
#SBATCH --job-name=deepseek-math-ul2-gsm8k-septoken-maskfull-sentence-equation-lossfulltarget-mixedcausalsenteqmasking-5ep-fullft
#SBATCH --output=/home/hieupq1/hieupq1/math/logs/slurm_%x.out
#SBATCH --error=/home/hieupq1/hieupq1/math/logs/slurm_%x.err
#SBATCH --nodes=2
#SBATCH --gpus-per-node=2
#SBATCH --mem-per-gpu=40GB
#SBATCH --cpus-per-gpu=16
#SBATCH --partition=applied
#SBATCH --mail-type=all
#SBATCH --mail-user=v.hieupq1@vinai.io
#SBATCH --dependency=116188
#SBATCH --exclude=sdc2-hpc-dgx-a100-001,sdc2-hpc-dgx-a100-002,sdc2-hpc-dgx-a100-009

JOB_NAME="deepseek-math-ul2-gsm8k-septoken-maskfull-sentence-equation-lossfulltarget-mixedcausalsenteqmasking-5ep-fullft"
output_dir="saves/${JOB_NAME}"
datasets="gsm8k_train_5_ul2_mixedcausalsenteqmasking"
ul2_causal=false
template="deepseek-math"
model_path="deepseek-math-7b-base"
n_epoch=1

module load python/miniconda3/miniconda3
eval "$(conda shell.bash hook)"
conda activate llama_fac
cd /home/hieupq1/hieupq1/math/

#Nodes
export GPUS_PER_NODE=2
export NUM_NODES=2
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=60001
# on your cluster you might need these:
# set the network interface
# export NCCL_SOCKET_IFNAME=^docker0,lo
export NCCL_SOCKET_IFNAME=bond0
export FI_EFA_USE_DEVICE_RDMA=1
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=WARN
export CUDA_LAUNCH_BLOCKING=0

srun --jobid $SLURM_JOBID bash -c "NCCL_DEBUG=INFO  python3 -m torch.distributed.run --nnodes $NUM_NODES --nproc_per_node $GPUS_PER_NODE --node_rank $SLURM_PROCID --master_addr $MASTER_ADDR --master_port $MASTER_PORT LLaMA-Factory/src/train.py \
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
  --gradient_accumulation_steps 16 \
  --ul2_finetune_embedding true \
  --ul2_causal $ul2_causal"
