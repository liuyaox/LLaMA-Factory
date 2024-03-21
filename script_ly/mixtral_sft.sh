#!/bin/bash
set -ex

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )   # cd到脚本所在目录，并返回绝对路径给SCRIPT_DIR
export DEPT_HOME=/data/user/ai_story
export LY_HOME=$DEPT_HOME/yao.liu
export RUN_ROOT=$LY_HOME/npc_llm

export NCCL_VERSION=2.17.1
export NCCL_IB_HCA=mlx5
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TIMEOUT=1000
export NCCL_IB_TC=136
export NCCL_IB_SL=5
export NCCL_IB_SPLIT_DATA_ON_QPS=1
export NCCL_IB_QPS_PER_CONNECTION=8
export NCCL_IB_RETRY_CNT=13
export NCCL_NET=Socket
export NCCL_SOCKET_IFNAME=bond0   # bond0仅香港集群有效，乌兰察布和火山是eth0（一般建议在DLC里跑训练任务，若使用DSW，需要注释这一行）
export NCCL_SHM_DISABLE=1
export NCCL_SOCKET_NTHREADS=4
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=10
export NCCL_DEBUG=INFO


per_device_train_batch_size=1
gradient_accumulation_steps=1
learning_rate=5e-6
num_train_epochs=5
seq_len=4096

num_machines=6
num_processes=48

RUN_GROUP=Mixtral_8x7B_Instruct_V0.1_SFT_LR${learning_rate}_EPOCH${num_train_epochs}_BS${per_device_train_batch_size}_SEQ${seq_len}_PROC${num_processes}
RUN_NAME=${RUN_GROUP}_$(date +%Y%m%d)
RUN_DIR=$RUN_ROOT/$RUN_NAME

mkdir -p $RUN_DIR
mkdir -p "${RUN_DIR}/checkpoints/"
mkdir -p "${RUN_DIR}/log/"
mkdir -p "${RUN_DIR}/wandb/"
cat $0 > $RUN_DIR/launch_script.sh

export WANDB_API_KEY=c3e85199a4ec8fcf33fe2fcbcf55f4f7d3ea20e9
wandb login --relogin $WANDB_API_KEY
export WANDB_ENTITY=littlecatx
export WANDB_PROJECT=npc_llm
export WANDB_GROUP=$RUN_GROUP   # TODO 没发生作用？后续研究下！
export WANDB_NAME=$RUN_NAME


acc_config_file="${RUN_DIR}/accelerate_config.yaml"
cat << EOF > ${acc_config_file}
compute_environment: LOCAL_MACHINE
distributed_type: DEEPSPEED
deepspeed_config:
  gradient_accumulation_steps: ${gradient_accumulation_steps}
  offload_optimizer_device: cpu
  offload_param_device: cpu
  zero3_init_flag: false
  zero3_save_16bit_model: true
  zero_stage: 3
main_training_function: main
num_machines: ${num_machines}
num_processes: ${num_processes}
machine_rank: 0
downcast_bf16: 'no'
mixed_precision: bf16
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
EOF

cd $LY_HOME/fork/LLaMA-Factory
#accelerate="/home/ai_story/anaconda3/envs/npc/bin/accelerate"
#${accelerate} launch --config_file ${acc_config_file} src/train_bash.py \
accelerate launch --machine_rank ${RANK} \
  --main_process_ip ${MASTER_ADDR} \
  --main_process_port ${MASTER_PORT} \
  --config_file ${acc_config_file} \
  src/train_bash.py \
  --model_name_or_path /maindata/data/user/ai_story/nlp_models/mistralai/Mixtral-8x7B-Instruct-v0.1/ \
  --output_dir ${RUN_DIR}/checkpoints \
  --dataset sft_roleplay_20240303 \
  --template mistral \
  --cutoff_len $seq_len \
  --preprocessing_num_workers 4 \
  --do_train \
  --stage sft \
  --finetuning_type full \
  --overwrite_cache \
  --per_device_train_batch_size $per_device_train_batch_size \
  --gradient_accumulation_steps $gradient_accumulation_steps \
  --learning_rate $learning_rate \
  --lr_scheduler_type cosine \
  --num_train_epochs $num_train_epochs \
  --report_to wandb \
  --logging_steps 5 \
  --save_steps 60 \
  --plot_loss \
  --bf16 \
  --flash_attn True \
  --do_eval \
  --val_size 0.1 \
  --evaluation_strategy steps \
  --eval_steps 25 \
  &> ${RUN_DIR}/log/$(date +%Y%m%d).log
