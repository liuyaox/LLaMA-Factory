#CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
#export CUDA_VISIBLE_DEVICES=0,1,2,3

export NCCL_NET=Socket
export NCCL_SOCKET_IFNAME=eth0
export NCCL_SHM_DISABLE=1
export NCCL_SOCKET_NTHREADS=4
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=10
export NCCL_DEBUG=INFO

accelerate launch src/train_bash.py \
  --stage sft \
  --do_train \
  --model_name_or_path  /mnt/model_zoo/Mixtral-8x7B-Instruct-v0.1/ \
  --dataset appen_1118 \
  --template mistral_sft \
  --finetuning_type full \
  --output_dir ./linky_mistral8x7b_appen1118_sft_full_v1p3/ \
  --overwrite_cache \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 1 \
  --lr_scheduler_type cosine \
  --logging_steps 10 \
  --save_steps 50 \
  --learning_rate 1e-5 \
  --num_train_epochs 4 \
  --plot_loss \
  --repetition_penalty 1.2 \
  --cutoff_len 4096 \
  --bf16 \
  --flash_attn True \
  --do_eval \
  --val_size 0.1 \
  --evaluation_strategy steps \
  --eval_steps 25

