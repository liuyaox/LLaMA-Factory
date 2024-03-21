#!/bin/bash
set -ex

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )   # cd到脚本所在目录，并返回绝对路径给SCRIPT_DIR
export DEPT_HOME=/data/user/ai_story
export LY_HOME=$DEPT_HOME/yao.liu
export RUN_ROOT=$LY_HOME/npc_llm


num_processes=8
per_device_train_batch_size=2
gradient_accumulation_steps=1
learning_rate=1e-6
num_train_epochs=5
seq_len=4096

MODEL_PATH=$RUN_ROOT/Mistral_7B_V0.1_CPT_DP16_VPPNone_ACC32_MBSZ2_GBSZ1024_SEQLEN4096_TRAIN_ITERS6000_20240223/checkpoints/npcllm_mistral7b_pt_ckpt3000_20240223

RUN_GROUP=Mistral_7B_V0.1_CPT_SFT_LR${learning_rate}_EPOCH${num_train_epochs}_BS${per_device_train_batch_size}_SEQ${seq_len}
RUN_NAME=${RUN_GROUP}_$(date +%Y%m%d)
RUN_DIR=$RUN_ROOT/$RUN_NAME

mkdir -p $RUN_DIR
mkdir -p "${RUN_DIR}/checkpoints/"
mkdir -p "${RUN_DIR}/log/"
mkdir -p "${RUN_DIR}/wandb/"
cat $0 > $RUN_DIR/launch_script.sh

export WANDB_API_KEY=c3e85199a4ec8fcf33fe2fcbcf55f4f7d3ea20e9
wandb login --relogin $WANDB_API_KEY
export WANDB_ENTITY=littlecatx  # 不指定的话，老是跑到ly_kunlun这个TEAM里
export WANDB_PROJECT=npc_llm
export WANDB_GROUP=$RUN_GROUP   # TODO 没发生作用？后续研究下！
export WANDB_NAME=$RUN_NAME

acc_config_file="${RUN_DIR}/accelerate_config.yaml"
cat << EOF > ${acc_config_file}
compute_environment: LOCAL_MACHINE
deepspeed_config:
  gradient_accumulation_steps: ${gradient_accumulation_steps}
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
num_machines: 1
num_processes: ${num_processes}
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
EOF

cd $LY_HOME/fork/LLaMA-Factory
accelerate="/home/ai_story/anaconda3/envs/npc/bin/accelerate"
#CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
${accelerate} launch --config_file ${acc_config_file} \
  src/train_bash.py \
  --model_name_or_path $MODEL_PATH \
  --dataset sft_roleplay_20240303 \
  --template mistral \
  --cutoff_len $seq_len \
  --preprocessing_num_workers 4 \
  --overwrite_cache \
  --do_train \
  --stage sft \
  --finetuning_type full \
  --per_device_train_batch_size $per_device_train_batch_size \
  --gradient_accumulation_steps $gradient_accumulation_steps \
  --learning_rate $learning_rate \
  --lr_scheduler_type cosine \
  --num_train_epochs $num_train_epochs \
  --bf16 \
  --flash_attn \
  --output_dir ${RUN_DIR}/checkpoints \
  --save_steps 60 \
  --report_to wandb \
  --logging_steps 5 \
  --plot_loss \
  --do_eval \
  --val_size 0.1 \
  --evaluation_strategy steps \
  --eval_steps 25 \
  &> ${RUN_DIR}/log/$(date +%Y%m%d).log
#    --ddp_timeout 7200
