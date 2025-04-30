# Reference Running: bash train/sft.sh
# {'train_runtime': 5268.8407, 'train_samples_per_second': 0.949, 'train_steps_per_second': 0.119, 'train_loss': 0.1172730620391667, 'epoch': 5.0}
export CUDA_VISIBLE_DEVICES=0,1
export HF_HOME="/data2/leczhang/models"

uid="$(date +%Y%m%d_%H%M%S)"
base_model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
lr=5e-6
min_lr=0
epochs=3
weight_decay=1e-4 # -> the same training pipe as slurm_training
micro_batch_size=2 # -> batch_size will be 16 if 16 gpus
gradient_accumulation_steps=8 # requires more GPU memory
max_steps=-1
# gpu_count=$(nvidia-smi -L | wc -l)
gpu_count=2
push_to_hub=false

torchrun --nproc-per-node ${gpu_count} --master_port 12345 \
    sft.py \
    --block_size=32768 \
    --per_device_train_batch_size=${micro_batch_size} \
    --per_device_eval_batch_size=${micro_batch_size} \
    --gradient_accumulation_steps=${gradient_accumulation_steps} \
    --num_train_epochs=${epochs} \
    --train_file_path="MATH_hard_train_proxy_tuning_cleaned.jsonl" \
    --model_name=${base_model} \
    --warmup_ratio=0.1 \
    --fsdp="full_shard auto_wrap" \
    --fsdp_config="fsdp_config_qwen.json" \
    --bf16=True \
    --eval_strategy="steps" \
    --eval_steps=25 \
    --logging_steps=1 \
    --save_strategy="steps" \
    --save_steps=25 \
    --lr_scheduler_type="cosine" \
    --learning_rate=${lr} \
    --weight_decay=${weight_decay} \
    --adam_beta1=0.9 \
    --adam_beta2=0.95 \
    --output_dir="/data2/leczhang/data-mixture/ckpts/sft-${uid}" \
    --push_to_hub=${push_to_hub} \
    --save_only_model=True
    # --gradient_checkpointing=True \ Enable gradient checkpointing for efficient memory usage with 8 H100 GPUs.
    # --accelerator_config='{"gradient_accumulation_kwargs": {"sync_each_batch": true}}'