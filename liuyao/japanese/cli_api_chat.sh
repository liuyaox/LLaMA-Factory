# Chat
JA_PATH=/maindata/data/user/ai_story/yao.liu/multilingual/Japanese
CKPT_PATH=/maindata/data/user/ai_story/zexu.sun/jap_sft/Qwen2-57B-A14B-Instruct_SFT_SEQ4096_LR1e-5_EP4_GBS32x1x1_NEFT0_20240701_synthetic0620/checkpoints/checkpoint-520
llamafactory-cli chat \
  --model_name_or_path ${CKPT_PATH} \
  --template qwen \
  --infer_backend vllm \
  --vllm_enforce_eager


# API
JA_PATH=/maindata/data/user/ai_story/yao.liu/multilingual/Japanese
CKPT_PATH=${JA_PATH}/Qwen2-57B-A14B-Instruct_SFT_SEQ4096_LR5e-6_EP4_GBS32x1x1_NEFT0_20240609_synthetic0530_common1/checkpoints/checkpoint-260
llamafactory-cli api \
  --model_name_or_path ${CKPT_PATH} \
  --template qwen \
  --infer_backend vllm \
  --vllm_enforce_eager

