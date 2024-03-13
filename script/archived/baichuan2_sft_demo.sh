# 来自：https://github.com/hiyouga/LLaMA-Factory/issues/2622

torchrun --nnodes $SLURM_JOB_NUM_NODES --master_addr $MASTER_ADDR --master_port $MASTER_PORT --node_rank $SLURM_PROCID --nproc_per_node 8 \
  src/train_bash.py \
  --stage sft \
  --finetuning_type full \
  --model_name_or_path ${model_name_or_path} \
  --deepspeed configs/deepspeed_configbf.json \
  --do_train \
  --save_strategy "steps" \
  --save_steps 1000 \
  --dataset_dir ${data_dir} \
  --dataset ${dataset} \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --preprocessing_num_workers 4 \
  --template baichuan2 \
  --num_train_epochs 8 \
  --cutoff_len 8192 \
  --save_strategy "steps" \
  --learning_rate 5e-5 \
  --max_grad_norm 1.0 \
  --weight_decay 1e-4 \
  --warmup_ratio 0.0 \
  --lr_scheduler_type "cosine" \
  --logging_steps 10 \
  --bf16 True \
  --tf32 True \
  --cache_dir ${cache_dir} \
  --output_dir ${output_dir} \
  --report_to none
