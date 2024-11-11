#!/bin/bash
set -ex

source /root/miniconda3/bin/activate lylf
export DEPT_HOME=/mnt/data
export LY_HOME=$DEPT_HOME/yao.liu
export RUN_ROOT=$LY_HOME/english/sfw


# -----------配置修改区------------
MODEL_PATH=/mnt/data/nlp_models/meta-llama/Meta-Llama-3-8B-Instruct
RUN_GROUP=Llama-3-8B-Instruct_SFT
TEMPLATE="llama3"

DATASET="en_sfw_online_erotic_1to1_dedup565,en_sfw_synthetic_v3_2_sfw_500,en_nsfw_synthetic_v3_2_nsfw_300,en_nsfw_synthetic_v3_2_normal_300,en_nsfw_15_1"
#TOTAL_SAMPLES=2321
V='v2_10'
TAG="both_five"

VAL_RATIO=0.05
EVAL_STEPS=4
LR=1e-6
EPOCHS=4
# -------------------------------


NUM_MACHINES=1
NUM_PROCESSES=8
SEQ_LEN=4096
PER_DEVICE_TRAIN_BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=4
#GLOBAL_BATCH_SIZE=$((NUM_PROCESSES * PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))  # 8 * 2 * 2 = 32
GLOBAL_BATCH_SIZE_STR=${NUM_PROCESSES}x${PER_DEVICE_TRAIN_BATCH_SIZE}x${GRADIENT_ACCUMULATION_STEPS}

#TRAIN_SAMPLES_PER_EPOCH=$(echo "scale=0; $TOTAL_SAMPLES * (1 - $VAL_RATIO) / 1" | bc)   # 1个epoch的训练样本数 5569*0.95 = 5291
#TRAIN_ITERS_PER_EPOCH=$((TRAIN_SAMPLES_PER_EPOCH / GLOBAL_BATCH_SIZE))                  # 1个epoch的迭代次数  5291/16 = 331 的确是这样！
#SAVE_STEPS=$((TRAIN_ITERS_PER_EPOCH / 1))    # 每个epoch保存1次   当只保存1次时，直接使用save_strategy=epoch吧
#EVAL_STEPS=$((TRAIN_ITERS_PER_EPOCH / 20))    # 每个epoch评估20次


RUN_NAME=${V}_${RUN_GROUP}_SEQ${SEQ_LEN}_LR${LR}_EP${EPOCHS}_GBS${GLOBAL_BATCH_SIZE_STR}_$(date +%Y%m%d)_${TAG}
RUN_DIR=$RUN_ROOT/$RUN_NAME

mkdir -p $RUN_DIR
mkdir -p "${RUN_DIR}/checkpoints/"
mkdir -p "${RUN_DIR}/log/"
mkdir -p "${RUN_DIR}/wandb/"
cat $0 > $RUN_DIR/launch_script.sh

export WANDB_API_KEY=c3e85199a4ec8fcf33fe2fcbcf55f4f7d3ea20e9
wandb login --relogin $WANDB_API_KEY
export WANDB_ENTITY=littlecatx
export WANDB_PROJECT=linky_english_sfw
export WANDB_GROUP=$RUN_GROUP
export WANDB_NAME=$RUN_NAME


ACC_CONFIG_FILE="${RUN_DIR}/accelerate_config.yaml"
cat << EOF > ${ACC_CONFIG_FILE}
compute_environment: LOCAL_MACHINE
distributed_type: DEEPSPEED
deepspeed_config:
  gradient_accumulation_steps: auto
  offload_optimizer_device: cpu
  offload_param_device: cpu
  zero3_init_flag: false
  zero3_save_16bit_model: true
  zero_stage: 3
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


cd $LY_HOME/fork/LLaMA-Factory
#/root/miniconda3/envs/lylf/bin/accelerate launch \
accelerate launch \
  --config_file ${ACC_CONFIG_FILE} \
  src/train.py \
  --model_name_or_path $MODEL_PATH \
  --dataset $DATASET \
  --template $TEMPLATE \
  --cutoff_len ${SEQ_LEN} \
  --preprocessing_num_workers 32 \
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
  --flash_attn auto \
  --output_dir ${RUN_DIR}/checkpoints \
  --save_strategy epoch \
  --report_to wandb \
  --run_name $RUN_NAME \
  --logging_steps 1 \
  --plot_loss \
  --ddp_timeout 7200 \
  --do_eval \
  --val_size $VAL_RATIO \
  --per_device_eval_batch_size 4 \
  --evaluation_strategy steps \
  --eval_steps $EVAL_STEPS \
  2>&1 | tee ${RUN_DIR}/log/$(date +%Y%m%d).log
