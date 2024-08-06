
# 一、基于chat/chat_model.py的ChatModel
# 支持参数：hparams/parser.py中_INFER_ARGS = [ModelArguments, DataArguments, FinetuningArguments, GeneratingArguments]
# 好像不支持指定模型加载的torch_dtype，其核心逻辑：bf16(若模型config中指定的是bf16) > fp16 > fp32

## 1. 结合FastAPI来调用API：支持批量，支持多卡(会自动分配多卡)     亲测成功
llamafactory-cli api liuyao/inference/cli_api_chat_webchat.yaml

MODEL_PATH=/maindata/data/shared/ai_story_workspace/yao.liu/english/normal/v1_5_Llama-3-8B-Instruct_SFT_SEQ4096_LR1e-5_EP6_GBS8x2x4_20240725_synthetic_v3_2_all_cleaned/checkpoints/checkpoint-139
llamafactory-cli api --model_name_or_path ${MODEL_PATH} \
  --template llama3 \
  --infer_backend vllm \
  --vllm_enforce_eager


## 2. 结合CLI来交互式聊天：不支持批量，支持多卡      亲测成功，2卡加载MOE有时会OOM
llamafactory-cli chat liuyao/inference/cli_api_chat_webchat.yaml


## 3. 结合WebUI来使用：不支持批量？支持多卡么？（其实使用的是ChatModel子类WebChatModel）    待以后可以端口转发了再测试下
llamafactory-cli webchat liuyao/inference/cli_api_chat_webchat.yaml


# 二、基于train/tuner.py中的run_exp函数
# 支持参数：hparams/parser.py中_TRAIN_ARGS = [ModelArguments, DataArguments, Seq2SeqTrainingArguments, FinetuningArguments, GeneratingArguments]
# 因为走的是train路径，多了Seq2SeqTrainingArguments，所以支持，比如bf16/fp16，会传递给model_args.compute_dtype，继而传递给load_model参数torch_dtype，若没指定，则逻辑同上面的inter路径

## 4. 直接运行脚本：支持批量，不支持多卡（不支持deepspeed）
llamafactory-cli train liuyao/inference/cli_train_do_predict.yaml   # 亲测成功
bash liuyao/inference/train_do_predict.sh


# 三、其他事项
# 1. 以上两种方式，背后加载模型都是靠load_model函数
# 2. 生成多个结果：修改seed，多次运行  https://github.com/hiyouga/LLaMA-Factory/issues/3820
# 3. [GREAT]训练集、推理集的数据构造与我相同：https://github.com/hiyouga/LLaMA-Factory/issues/2794

# 关于多卡
# do_predict不支持deepspeed，建议使用api_demo+vllm  https://github.com/hiyouga/LLaMA-Factory/issues/3191
# 使用api_demo会自动分配多卡，predict不提供多卡推理能力，建议使用vLLM  https://github.com/hiyouga/LLaMA-Factory/issues/2318
