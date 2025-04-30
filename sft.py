import os
from dataclasses import dataclass, field, asdict
from typing import Optional
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from datasets import load_dataset, concatenate_datasets, DatasetDict, Dataset
import transformers
import trl
import pandas as pd


# Extract the boxed answer from the model output
def extract_boxed_answer(text):
    start_token = r"\boxed{"
    start_idx = text.rfind(start_token)
    if start_idx == -1:
        return ''
    i = start_idx + len(start_token)
    brace_depth = 1
    content = []
    while i < len(text):
        if text[i] == '{':
            brace_depth += 1
        elif text[i] == '}':
            brace_depth -= 1
            if brace_depth == 0:
                break
        content.append(text[i])
        i += 1
    return ''.join(content).strip() if brace_depth == 0 else ''

@dataclass
class TrainingConfig:
    model_name: str = field(default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    block_size: int = field(default=32768)
    wandb_project: Optional[str] = field(default="long-cot-labwide")
    wandb_entity: Optional[str] = field(default="orange0629")
    train_file_path: Optional[str] = field(default='simplescaling/s1K-1.1')
    dagger: bool = field(default=False)

    def __post_init__(self):
        os.environ['WANDB_PROJECT'] = self.wandb_project
        os.environ['WANDB_ENTITY'] = self.wandb_entity

def train():
    # parsing input
    parser = transformers.HfArgumentParser((TrainingConfig, trl.SFTConfig))
    config, args = parser.parse_args_into_dataclasses()
    log_config = {**asdict(config), **asdict(args)}
    logging.info(f"Training config: {log_config}")

    # loading model
    kwargs = {}
    if "70B" in config.model_name:
        # Removed "low_cpu_mem_usage": True, for 70B, since by default we are in FSDP,
        # it's more efficient to do  "cpu_ram_efficient_loading": true, in fsdp_config.json
        kwargs = {"device_map": "auto", "torch_dtype": "auto",
                  "attn_implementation": "flash_attention_2", "use_cache": False}
        model = transformers.AutoModelForCausalLM.from_pretrained(config.model_name, **kwargs)
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(config.model_name)

    # 1. Read jsonl
    df = pd.read_json(config.train_file_path, lines=True)
    train_df = df[df['split'] == 'train'].reset_index(drop=True)
    dev_df = df[df['split'] == 'dev'].reset_index(drop=True)

    # 3. 转成Dataset
    train_dataset = Dataset.from_pandas(train_df)
    dev_dataset = Dataset.from_pandas(dev_df)

    def apply_chat_template(example):
        cot_prefix = "Solve the following math problem. Put your final answer within \\boxed{}.\n\n"
        return {
            "text": f"<｜begin▁of▁sentence｜><｜User｜>{cot_prefix}Question: {example['question']}<｜Assistant｜>{example['model_output_final']}<｜end▁of▁sentence｜>",
            "input_text_only": f"<｜begin▁of▁sentence｜><｜User｜>{cot_prefix}Question: {example['question']}<｜Assistant｜>",
        }
    train_dataset = train_dataset.map(apply_chat_template)
    dev_dataset = dev_dataset.map(apply_chat_template)

    # setting up trainer
    tokenizer = transformers.AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    if "DeepSeek-R1" in config.model_name:
        instruction_template = "<｜begin▁of▁sentence｜><｜User｜>"
        response_template = "<｜Assistant｜>"
    elif "Llama" in config.model_name:
        instruction_template = "<|start_header_id|>user<|end_header_id|>"
        response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    elif "Qwen" in config.model_name:
        instruction_template = "<|im_start|>user"
        response_template = "<|im_start|>assistant\n"

    args.save_total_limit = None

    # Only compute loss over assistant responses
    # Verified that it precisely starts where the thinking tokens start and ends with the first pad token
    # via labels being set to -100
    collator = trl.DataCollatorForCompletionOnlyLM(
        instruction_template=instruction_template,
        response_template=response_template,
        tokenizer=tokenizer,
        mlm=False
    )
    args.dataset_text_field = 'text'
    args.max_seq_length = config.block_size
    trainer = trl.SFTTrainer(
        model,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        args=args,
        data_collator=collator,
    )

    trainer.train()
    # trainer.save_model(output_dir=args.output_dir)
    # tokenizer.save_pretrained(args.output_dir)
    trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
    train()