#!/bin/bash

set -ex
# export NCCL_IB_HCA=mlx5
# export NCCL_IB_TC=136
# export NCCL_IB_SL=5
# export NCCL_IB_GID_INDEX=3
# export NCCL_SOCKET_IFNAME=bond0
# export NCCL_DEBUG=INFO

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
# 使用dsw启动时，需要将下面这行注释
export NCCL_SOCKET_IFNAME=bond0     #仅香港集群有效， 乌兰察布 和 火山 是  eth0
export NCCL_DEBUG=INFO


#CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \

# CUDA_VISIBLE_DEVICES="0,1"


LLaMA_FACTORY_PATH="/data/user/ai_story/David/LLaMA-Factory"
MODEL_PATH="/maindata/data/user/ai_story/nlp_models/mistralai/Mixtral-8x7B-v0.1"
accelerate="/data/user/ai_story/David/miniconda/envs/storyllm/bin/accelerate"


#alias deepspeed='/data/user/ai_story/David/miniconda/envs/storyllm/bin/deepspeed'



gradient_accumulation_steps=1

# 获取当前文件所在的文件夹绝对路径
current_folder=$(dirname "$(readlink -f "$0")")
acc_config_file_path="${current_folder}/accelerate_config.yaml"


cat << EOF > $acc_config_file_path
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
num_machines: 4
num_processes: 32
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
EOF


# 设置脚本和脚本参数
script="src/train_bash.py"

script_args="--stage sft \
    --model_name_or_path ${MODEL_PATH} \
    --do_train \
    --dataset story_sft_v1 \
    --template storyllm \
    --finetuning_type full \
    --preprocessing_num_workers 4 \
    --output_dir /maindata/data/user/ai_story/David/sft_models/mixtral-8x7b-sft_v04_fsdp \
    --overwrite_cache \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 3200 \
    --learning_rate 5e-6 \
    --num_train_epochs 2 \
    --plot_loss \
    --cutoff_len 4096 \
    --bf16 \
    --flash_attn \
    --ddp_timeout 7200
"

# 组装执行命令
accelerate_command="${accelerate} launch --machine_rank ${RANK} --main_process_ip ${MASTER_ADDR} --main_process_port ${MASTER_PORT} --config_file=${acc_config_file_path} $script $script_args"
echo $accelerate_command

# 执行accelerate命令
cd $LLaMA_FACTORY_PATH
$accelerate_command
