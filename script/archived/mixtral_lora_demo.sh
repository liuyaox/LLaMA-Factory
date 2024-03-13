# LLaMA-Factory作者提供，来自https://huggingface.co/mistralai/Mixtral-8x7B-v0.1/discussions/10
CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
  --stage sft \
  --do_train \
  --model_name_or_path mistralai/Mixtral-8x7B-v0.1 \
  --dataset alpaca_en \
  --template mistral \
  --finetuning_type lora \
  --lora_target q_proj,v_proj \
  --output_dir mixtral \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --lr_scheduler_type cosine \
  --logging_steps 10 \
  --save_steps 1000 \
  --learning_rate 5e-5 \
  --num_train_epochs 1.0 \
  --quantization_bit 4 \
  --bf16


# 来自https://blog.devgenius.io/how-to-fine-tune-mixtral-8x7b-instruct-on-your-own-data-78f3b2f8c808
# 与上面的参数一模一样，除了模型和数据
CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
  --stage sft \
  --do_train \
  --model_name_or_path mistralai/Mixtral-8x7B-Instruct-v0.1 \
  --dataset example \
  --template mistral \
  --finetuning_type lora \
  --lora_target q_proj,v_proj \
  --output_dir mixtral \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --lr_scheduler_type cosine \
  --logging_steps 10 \
  --save_steps 1000 \
  --learning_rate 5e-5 \
  --num_train_epochs 1.0 \
  --quantization_bit 4 \
  --bf16
