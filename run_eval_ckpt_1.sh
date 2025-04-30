export CUDA_VISIBLE_DEVICES=0
export HF_HOME="/data2/leczhang/models"

python eval_checkpoints.py --model_dir "/data2/leczhang/data-mixture/ckpts/sft-20250430_121447/checkpoint-25"
python eval_checkpoints.py --model_dir "/data2/leczhang/data-mixture/ckpts/sft-20250430_121447/checkpoint-50"
python eval_checkpoints.py --model_dir "/data2/leczhang/data-mixture/ckpts/sft-20250430_121447/checkpoint-75"
python eval_checkpoints.py --model_dir "/data2/leczhang/data-mixture/ckpts/sft-20250430_121447/checkpoint-100"
python eval_checkpoints.py --model_dir "/data2/leczhang/data-mixture/ckpts/sft-20250430_121447/checkpoint-125"