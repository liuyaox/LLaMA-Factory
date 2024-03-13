#!/bin/bash
set -ex

pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
pip config set install.trusted-host mirrors.aliyun.com
pip install wandb
pip install tiktoken
pip install transformers==4.36.2  # 避免多节点保存checkpoint时的错误

export DEPT_HOME=/maindata/data/user/ai_story
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
export NCCL_NET_PLUGIN=none
export NCCL_SOCKET_IFNAME=bond0     #仅香港集群有效， 乌兰察布 和 火山 是  eth0   # 使用dsw启动时，需要将下面这行注释
export NCCL_DEBUG=INFO


learning_rate=5e-6
num_train_epochs=4
per_device_train_batch_size=1
gradient_accumulation_steps=1
seq_len=4096
num_machines=4
num_processes=32

#MODEL_PATH=$DEPT_HOME/nlp_models/mistralai/Mixtral-8x7B-v0.1
MODEL_PATH=$DEPT_HOME/nlp_models/mistralai/Mixtral-8x7B-Instruct-v0.1

#RUN_GROUP=Mixtral_8x7B_v0.1_SFT_LR${learning_rate}_EPOCH${num_train_epochs}_BS${per_device_train_batch_size}_SEQ${seq_len}_PROC${num_processes}
RUN_GROUP=Mixtral_8x7B_Instruct_v0.1_SFT_LR${learning_rate}_EPOCH${num_train_epochs}_BS${per_device_train_batch_size}_SEQ${seq_len}_PROC${num_processes}
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
export WANDB_GROUP=$RUN_GROUP
export WANDB_NAME=$RUN_NAME

acc_config_file="${RUN_DIR}/accelerate_config.yaml"
cat << EOF > $acc_config_file
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: FSDP
downcast_bf16: 'no'
fsdp_config:
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_backward_prefetch: BACKWARD_PRE
  fsdp_cpu_ram_efficient_loading: true
  fsdp_forward_prefetch: false
  fsdp_offload_params: false
  fsdp_sharding_strategy: 1
  fsdp_state_dict_type: FULL_STATE_DICT
  fsdp_sync_module_states: true
  fsdp_use_orig_params: true
main_training_function: main
mixed_precision: bf16
num_machines: ${num_machines}
num_processes: ${num_processes}
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
EOF

#cd $LY_HOME/fork/LLaMA-Factory
cd /data/user/ai_story/David/LLaMA-Factory
#accelerate="/data/user/ai_story/David/miniconda/envs/storyllm/bin/accelerate"
#${accelerate} launch --machine_rank ${RANK} \
accelerate launch --machine_rank ${RANK} \
  --main_process_ip ${MASTER_ADDR} \
  --main_process_port ${MASTER_PORT} \
  --config_file=${acc_config_file} \
  src/train_bash.py \
  --model_name_or_path $MODEL_PATH \
  --dataset sft_roleplay_20240303 \
  --template mistral \
  --cutoff_len ${seq_len} \
  --preprocessing_num_workers 4 \
  --overwrite_cache \
  --do_train \
  --stage sft \
  --finetuning_type full \
  --per_device_train_batch_size ${per_device_train_batch_size} \
  --gradient_accumulation_steps ${gradient_accumulation_steps} \
  --learning_rate ${learning_rate} \
  --lr_scheduler_type cosine \
  --num_train_epochs ${num_train_epochs} \
  --bf16 \
  --flash_attn \
  --output_dir ${RUN_DIR}/checkpoints \
  --save_steps 61 \
  --report_to wandb \
  --logging_steps 5 \
  --plot_loss \
  --ddp_timeout 7200 \
  --do_eval \
  --val_size 0.1 \
  --evaluation_strategy steps \
  --eval_steps 25 \
  2>&1 | tee ${RUN_DIR}/log/$(date +%Y%m%d).log
