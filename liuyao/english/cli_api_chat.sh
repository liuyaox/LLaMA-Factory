EN_PATH=/maindata/data/shared/ai_story_workspace/yao.liu/english/normal
CKPT_PATH=${EN_PATH}/v1_0_Llama-3-8B-Instruct_SFT_SEQ4096_LR1e-5_EP4_GBS8x2x4_20240717_synthetic_v3_2_all/checkpoints/checkpoint-328

# Chat
llamafactory-cli chat \
  --model_name_or_path ${CKPT_PATH} \
  --template llama3 \
  --infer_backend vllm \
  --vllm_enforce_eager

# API
llamafactory-cli api \
  --model_name_or_path ${CKPT_PATH} \
  --template llama3 \
  --infer_backend vllm \
  --vllm_enforce_eager
