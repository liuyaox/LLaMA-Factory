#!/bin/bash
set -ex

pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
pip config set install.trusted-host mirrors.aliyun.com
pip install trl==0.7.4

export DEPT_HOME=/maindata/data/user/ai_story
export LY_HOME=$DEPT_HOME/yao.liu
export RUN_ROOT=$LY_HOME/multilingual/Japanese


# -----------配置修改区------------
MODEL_PATH=/maindata/data/shared/public/ai_story/nlp_models/mistralai/Mixtral-8x7B-Instruct-v0.1
DATASET="japanese_synthetic_0516_mixtral8x7b"
TOTAL_SAMPLES=26481
RUN_GROUP=Mixtral-8x7B-Instruct-v0.1_SFT
TAG="synthetic_0516_deepspeed"
# -------------------------------


NUM_MACHINES=4
NUM_PROCESSES=32
EPOCHS=4
LR=5e-6
SEQ_LEN=4096
NEFTUNE_NOISE_ALPHA=0
PER_DEVICE_TRAIN_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=1
# TODO 好像有问题！？！下面的NUM_PROCESSES，本质上应该是DP，而NUM_PROCESSES其实=DP*PP*TP ！
# sft_id.sh中：在2节点16卡，acc=1时，GLOBAL好像是8，而不是16*1*1，说明这里DP=8，也就是说，PP*TP=2（自动拆分的？）
# sft.sh中：在3节点24卡，acc=2时，共有724个step，说明GLOBAL好像是16，而不是24*2*1，说明这里DP=8，也就是说，PP*TP=3（自动拆分的？）
GLOBAL_BATCH_SIZE=$((NUM_PROCESSES * PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))  # 8 * 1 * 2 = 16
GLOBAL_BATCH_SIZE_STR=${NUM_PROCESSES}x${PER_DEVICE_TRAIN_BATCH_SIZE}x${GRADIENT_ACCUMULATION_STEPS}

VAL_RATIO=0.05
TRAIN_SAMPLES_PER_EPOCH=$(echo "scale=0; $TOTAL_SAMPLES * (1 - $VAL_RATIO) / 1" | bc)   # 1个epoch的训练样本数
TRAIN_ITERS_PER_EPOCH=$((TRAIN_SAMPLES_PER_EPOCH / GLOBAL_BATCH_SIZE))                  # 1个epoch的迭代次数    93   187
#SAVE_STEPS=$((TRAIN_ITERS_PER_EPOCH / 1))    # 每个epoch保存1次   当只保存1次时，直接使用save_strategy=epoch吧
#EVAL_STEPS=$((TRAIN_ITERS_PER_EPOCH / 20))    # 每个epoch评估20次
EVAL_STEPS=100


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
cat << EOF > ${ACC_CONFIG_FILE}
compute_environment: LOCAL_MACHINE
deepspeed_config:
  gradient_accumulation_steps: ${GRADIENT_ACCUMULATION_STEPS}
  offload_optimizer_device: cpu
  offload_param_device: cpu
  zero3_init_flag: false
  zero3_save_16bit_model: true
  zero_stage: 3
distributed_type: DEEPSPEED
downcast_bf16: 'no'
machine_rank: 0
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

cd $LY_HOME/fork2/LLaMA-Factory
accelerate launch --machine_rank ${RANK} \
  --main_process_ip ${MASTER_ADDR} \
  --main_process_port ${MASTER_PORT} \
  --config_file ${ACC_CONFIG_FILE} \
  src/train_bash.py \
  --model_name_or_path $MODEL_PATH \
  --dataset $DATASET \
  --template custom_blank \
  --cutoff_len ${SEQ_LEN} \
  --preprocessing_num_workers 12 \
  --overwrite_cache \
  --do_train \
  --stage sft \
  --finetuning_type full \
  --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
  --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
  --learning_rate ${LR} \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.03 \
  --num_train_epochs ${EPOCHS} \
  --bf16 \
  --flash_attn \
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