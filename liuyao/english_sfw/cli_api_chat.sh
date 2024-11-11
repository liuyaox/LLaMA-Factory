VLLM_WORKER_MULTIPROC_METHOD=spawn
export CUDA_VISIBLE_DEVICES=1
PROJ_PATH=/mnt/data/yao.liu/english/sfw
CKPT_PATH=${PROJ_PATH}/v2_10_Llama-3-8B-Instruct_SFT_SEQ4096_LR1e-6_EP4_GBS8x2x4_20240819_both_five/checkpoints/checkpoint-104
CKPT_PATH=${PROJ_PATH}/v2_11_Llama-3-8B-Instruct_SFT_SEQ4096_LR5e-6_EP4_GBS8x2x4_20240819_both_five/checkpoints/checkpoint-52

# Chat
llamafactory-cli chat \
  --model_name_or_path ${CKPT_PATH} \
  --template llama3 \
  --infer_backend vllm \
  --vllm_enforce_eager \
  --vllm_maxlen 4096

# API
llamafactory-cli api \
  --model_name_or_path ${CKPT_PATH} \
  --template llama3 \
  --infer_backend vllm \
  --vllm_enforce_eager \
  --vllm_maxlen 4096


export CUDA_VISIBLE_DEVICES=2
PROJ_PATH=/mnt/data/yao.liu/english/sfw
CKPT_PATH=${PROJ_PATH}/v2_11_Llama-3-8B-Instruct_SFT_SEQ4096_LR5e-6_EP4_GBS8x2x4_20240819_both_five/checkpoints/checkpoint-52
llamafactory-cli api \
  --model_name_or_path ${CKPT_PATH} \
  --template llama3 \
  --infer_backend vllm \
  --vllm_enforce_eager \
  --vllm_maxlen 4096


curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "gpt-3.5-turbo",
        "prompt": "San Francisco is a",
        "max_tokens": 7,
        "temperature": 0
    }'