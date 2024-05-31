#!/bin/bash
set -ex

pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
pip config set install.trusted-host mirrors.aliyun.com
pip install trl==0.8.6
pip install peft==0.10.0
pip install transformers==4.40.0

export DEPT_HOME=/maindata/data/user/ai_story
export LY_HOME=$DEPT_HOME/yao.liu
export RUN_ROOT=$LY_HOME/multilingual/Japanese

export NCCL_VERSION=2.17.1
export NCCL_IB_HCA=mlx5
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TIMEOUT=10000    # 若报错socket timeout之类的错，可以设置大一些（待验证）？
export NCCL_IB_TC=136
export NCCL_IB_SL=5
export NCCL_IB_SPLIT_DATA_ON_QPS=1
export NCCL_IB_QPS_PER_CONNECTION=8
export NCCL_IB_RETRY_CNT=20
export NCCL_NET_PLUGIN=none
export NCCL_SOCKET_IFNAME=bond0
export NCCL_DEBUG=INFO


# -----------配置修改区------------
MODEL_PATH=/maindata/data/shared/public/ai_story/nlp_models/Japanese/karakuri-ai/karakuri-lm-8x7b-chat-v0.1
RUN_GROUP=karakuri-lm-8x7b-chat-v0.1_SFT

# 20240530
DATASET="japanese_synthetic_0530_karakuri_lm8x7b_chat,japanese_translate_0529_karakuri_lm8x7b_chat"
TOTAL_SAMPLES=90987
TAG="synthetic0530_translate0529"
VAL_RATIO=0.05
# -------------------------------


NUM_MACHINES=4      # 最低2个节点
NUM_PROCESSES=32
EPOCHS=4
LR=5e-6
SEQ_LEN=4096
NEFTUNE_NOISE_ALPHA=0
PER_DEVICE_TRAIN_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=1   # 2会OOM
GLOBAL_BATCH_SIZE=$((NUM_PROCESSES * PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))  # 16 * 1 * 1 = 16
GLOBAL_BATCH_SIZE_STR=${NUM_PROCESSES}x${PER_DEVICE_TRAIN_BATCH_SIZE}x${GRADIENT_ACCUMULATION_STEPS}

TRAIN_SAMPLES_PER_EPOCH=$(echo "scale=0; $TOTAL_SAMPLES * (1 - $VAL_RATIO) / 1" | bc)   # 1个epoch的训练样本数 5525
TRAIN_ITERS_PER_EPOCH=$((TRAIN_SAMPLES_PER_EPOCH / GLOBAL_BATCH_SIZE))    # 1个epoch的迭代次数   345  406
#SAVE_STEPS=$((TRAIN_ITERS_PER_EPOCH / 1))     # 每个epoch保存1次   当只保存1次时，直接使用save_strategy=epoch吧
#EVAL_STEPS=$((TRAIN_ITERS_PER_EPOCH / 20))    # 每个epoch评估20次
EVAL_STEPS=50


RUN_NAME=${RUN_GROUP}_SEQ${SEQ_LEN}_LR${LR}_EP${EPOCHS}_GBS${GLOBAL_BATCH_SIZE_STR}_NEFT${NEFTUNE_NOISE_ALPHA}_$(date +%Y%m%d)_${TAG}
RUN_DIR=$RUN_ROOT/$RUN_NAME

mkdir -p $RUN_DIR
mkdir -p "${RUN_DIR}/checkpoints/"
mkdir -p "${RUN_DIR}/log/"
mkdir -p "${RUN_DIR}/wandb/"
cat $0 > $RUN_DIR/launch_script.sh

export WANDB_API_KEY=c3e85199a4ec8fcf33fe2fcbcf55f4f7d3ea20e9
wandb login --relogin $WANDB_API_KEY
export WANDB_ENTITY=littlecatx
export WANDB_PROJECT=linky_llm_japanese
export WANDB_GROUP=$RUN_GROUP
export WANDB_NAME=$RUN_NAME


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


cd $LY_HOME/fork/LLaMA-Factory
accelerate launch --machine_rank ${RANK} \
  --main_process_ip ${MASTER_ADDR} \
  --main_process_port ${MASTER_PORT} \
  --config_file ${ACC_CONFIG_FILE} \
  src/train.py \
  --model_name_or_path $MODEL_PATH \
  --dataset ${DATASET} \
  --template empty \
  --cutoff_len ${SEQ_LEN} \
  --preprocessing_num_workers 16 \
  --overwrite_cache \
  --do_train \
  --stage sft \
  --finetuning_type full \
  --per_device_train_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
  --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
  --learning_rate ${LR} \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.03 \
  --num_train_epochs ${EPOCHS} \
  --bf16 \
  --flash_attn auto \
  --repetition_penalty 1.08 \
  --output_dir ${RUN_DIR}/checkpoints \
  --save_strategy epoch \
  --report_to wandb \
  --logging_steps 1 \
  --plot_loss \
  --ddp_timeout 7200 \
  --do_eval \
  --val_size $VAL_RATIO \
  --evaluation_strategy steps \
  --eval_steps $EVAL_STEPS \
  2>&1 | tee ${RUN_DIR}/log/$(date +%Y%m%d).log
#  --neftune_noise_alpha ${NEFTUNE_NOISE_ALPHA} \
