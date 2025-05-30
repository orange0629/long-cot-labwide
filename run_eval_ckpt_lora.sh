export CUDA_VISIBLE_DEVICES=1
export HF_HOME="/data2/leczhang/models"

python eval_checkpoints.py --model_dir "/data2/leczhang/data-mixture/ckpts/LLaMA-Factory/saves/ds_r1_qwen_1.5B/full/lora/checkpoint-50" --lora
python eval_checkpoints.py --model_dir "/data2/leczhang/data-mixture/ckpts/LLaMA-Factory/saves/ds_r1_qwen_1.5B/full/lora/checkpoint-100" --lora
