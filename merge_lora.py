import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["HF_HOME"] = "/data2/leczhang/models"
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

# Step 1: Load PEFT config
peft_model_path = "/data2/leczhang/data-mixture/ckpts/LLaMA-Factory/saves/ds_r1_qwen_1.5B/full/lora_qwen32b/checkpoint-100-acc-0.4800"
config = PeftConfig.from_pretrained(peft_model_path)

# Step 2: Load base model
base_model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, device_map="auto")

# Step 3: Load LoRA and merge
model = PeftModel.from_pretrained(base_model, peft_model_path)
model = model.merge_and_unload()  # This merges LoRA into base weights

# Step 4: Save the merged model
save_path = peft_model_path + "_merged"
model.save_pretrained(save_path)

# (Optional) Save tokenizer too
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
tokenizer.save_pretrained(save_path)
