#!/bin/bash
set -ex

pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
pip config set install.trusted-host mirrors.aliyun.com
pip install trl==0.8.6
pip install peft==0.10.0
pip install transformers==4.40.0

export DEPT_HOME=/maindata/data/user/ai_story
export LY_HOME=$DEPT_HOME/yao.liu
MODEL_PATH=$LY_HOME/multilingual/Japanese/suzume-llama-3-8B-japanese_SFT_SEQ4096_LR5e-6_EP4_GBS8x1x2_NEFT0_20240519_synthetic_0516/checkpoints/checkpoint-1573
OUTPUT_PATH=$LY_HOME/xxx

cd $LY_HOME/fork/LLaMA-Factory
#accelerate launch src/train.py \
CUDA_VISIBLE_DEVICES=0 \
python src/train.py \
    --model_name_or_path $MODEL_PATH \
    --bf16 \
    --dataset japanese_synthetic_0516_suzume-llama3-8b \
    --template empty \
    --cutoff_len 4096 \
    --max_samples 20 \
    --max_new_tokens 512 \
    --preprocessing_num_workers 16 \
    --overwrite_cache \
    --do_predict \
    --stage sft \
    --finetuning_type full \
    --per_device_eval_batch_size 8 \
    --predict_with_generate \
    --output_dir $OUTPUT_PATH \
    --overwrite_output_dir


# 串行跑多份数据
datasets=("hq_dense_top_2" "hq_dense_top_3_120")

for ((i=0; i<${#datasets[@]}; i++)); do
    dataset=${datasets[$i]}
    OUTPUT_PATH="xxx_$dataset"

    CUDA_VISIBLE_DEVICES=0 \
    python src/train.py \
        --xxx
done
