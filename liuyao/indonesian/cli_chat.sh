ID_PATH=/maindata/data/shared/ai_story_workspace/yao.liu/multilingual/Indonesian/v0_9_Sailor-7B-Chat_SFT_SEQ4096_LR5e-6_EP4_GBS8x2x4_20240723_expand_trans_write_chatedit/checkpoints
llamafactory-cli chat \
  --model_name_or_path ${ID_PATH}/checkpoint-354 \
  --template qwen \
  --infer_backend vllm \
  --vllm_enforce_eager
