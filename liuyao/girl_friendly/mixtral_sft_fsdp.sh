#!/bin/bash
set -ex

pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
pip config set install.trusted-host mirrors.aliyun.com
pip install wandb
pip install tiktoken
pip install transformers==4.36.2  # 避免多节点保存checkpoint时的错误
#pip install git+https://github.com/huggingface/transformers
#cd /maindata/data/user/ai_story/yao.liu/github/transformers
#pip install -e .


export DEPT_HOME=/maindata/data/user/ai_story
export LY_HOME=$DEPT_HOME/yao.liu
export RUN_ROOT=$LY_HOME/girl_friendly

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
export NCCL_SOCKET_IFNAME=bond0     #仅香港集群有效， 乌兰察布和火山是eth0   # 使用dsw启动时，需要将下面这行注释
export NCCL_DEBUG=INFO


NUM_TRAIN_EPOCHS=4
LEARNING_RATE=5e-6
SEQ_LEN=4096
NUM_MACHINES=2
NUM_PROCESSES=16
PER_DEVICE_TRAIN_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=1
GLOBAL_BATCH_SIZE=$((PER_DEVICE_TRAIN_BATCH_SIZE * NUM_PROCESSES * GRADIENT_ACCUMULATION_STEPS))  # 1 * 16 * 1 = 16

DATASET="girl_friendly_2.0"
#TOTAL_SAMPLES=1399  # 15.1
TOTAL_SAMPLES=5816  # 2.0
#TOTAL_SAMPLES=7215  # v1_merged
FLAG="2.0"

VAL_RATIO=0.1
TRAIN_SAMPLES_PER_EPOCH=5234  # 1259  5234  6494
#TRAIN_SAMPLES_PER_EPOCH=$((TOTAL_SAMPLES * (1 - VAL_RATIO)))             # 1个epoch的训练样本数 5525
TRAIN_ITERS_PER_EPOCH=$((TRAIN_SAMPLES_PER_EPOCH / GLOBAL_BATCH_SIZE))    # 1个epoch的迭代次数   345  406

SAVE_STEPS=$((TRAIN_ITERS_PER_EPOCH / 1))     # 每个epoch保存1次
EVAL_STEPS=$((TRAIN_ITERS_PER_EPOCH / 20))    # 每个epoch评估20次

#MODEL_PATH=$DEPT_HOME/nlp_models/mistralai/Mixtral-8x7B-v0.1
MODEL_PATH=$DEPT_HOME/nlp_models/mistralai/Mixtral-8x7B-Instruct-v0.1

#RUN_GROUP=Mixtral_8x7B_V0.1_SFT
RUN_GROUP=Mixtral_8x7B_Instruct_V0.1_SFT
RUN_NAME=${RUN_GROUP}_LR${LEARNING_RATE}_EPOCH${NUM_TRAIN_EPOCHS}_GBS${GLOBAL_BATCH_SIZE}_SEQ${SEQ_LEN}_PROC${NUM_PROCESSES}_$(date +%Y%m%d)_v-${FLAG}
RUN_DIR=$RUN_ROOT/$RUN_NAME

mkdir -p $RUN_DIR
mkdir -p "${RUN_DIR}/checkpoints/"
mkdir -p "${RUN_DIR}/log/"
mkdir -p "${RUN_DIR}/wandb/"
cat $0 > $RUN_DIR/launch_script.sh

export WANDB_API_KEY=c3e85199a4ec8fcf33fe2fcbcf55f4f7d3ea20e9
wandb login --relogin $WANDB_API_KEY
export WANDB_ENTITY=littlecatx
export WANDB_PROJECT=linky_girl_friendly
export WANDB_GROUP=$RUN_GROUP
export WANDB_NAME=$RUN_NAME
export WANDB_WATCH=all


ACC_CONFIG_FILE="${RUN_DIR}/accelerate_config.yaml"
cat << EOF > $ACC_CONFIG_FILE
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
num_machines: ${NUM_MACHINES}
num_processes: ${NUM_PROCESSES}
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
EOF

#cd $LY_HOME/fork/LLaMA-Factory
cd /maindata/data/user/ai_story/David/LLaMA-Factory
#accelerate="/data/user/ai_story/David/miniconda/envs/storyllm/bin/accelerate"
#${accelerate} launch --machine_rank ${RANK} \
accelerate launch --machine_rank ${RANK} \
  --main_process_ip ${MASTER_ADDR} \
  --main_process_port ${MASTER_PORT} \
  --config_file=${ACC_CONFIG_FILE} \
  src/train_bash.py \
  --model_name_or_path $MODEL_PATH \
  --dataset ${DATASET} \
  --template mistral_linky_preprocessed \
  --cutoff_len ${SEQ_LEN} \
  --preprocessing_num_workers 16 \
  --overwrite_cache \
  --do_train \
  --stage sft \
  --finetuning_type full \
  --per_device_train_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
  --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
  --learning_rate ${LEARNING_RATE} \
  --lr_scheduler_type cosine \
  --num_train_epochs ${NUM_TRAIN_EPOCHS} \
  --bf16 \
  --flash_attn \
  --output_dir ${RUN_DIR}/checkpoints \
  --save_steps $SAVE_STEPS \
  --report_to wandb \
  --logging_steps 1 \
  --plot_loss \
  --ddp_timeout 7200 \
  --do_eval \
  --val_size $VAL_RATIO \
  --evaluation_strategy steps \
  --eval_steps $EVAL_STEPS \
  2>&1 | tee ${RUN_DIR}/log/$(date +%Y%m%d).log
