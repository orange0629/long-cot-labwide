import os
import argparse
from tqdm import tqdm
import multiprocessing
from vllm import LLM, SamplingParams
import json
from datasets import load_dataset, concatenate_datasets, DatasetDict, Dataset
import pandas as pd
from eval.math_equivalence import is_equiv

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

def apply_chat_template(example):
    cot_prefix = "Solve the following math problem. Put your final answer within \\boxed{}.\n\n"
    return {
        "text": f"<｜begin▁of▁sentence｜><｜User｜>{cot_prefix}Question: {example['question']}<｜Assistant｜>{example['model_output_final']}<｜end▁of▁sentence｜>",
        "input_text_only": f"<｜begin▁of▁sentence｜><｜User｜>{cot_prefix}Question: {example['question']}<｜Assistant｜>",
    }

def main(model_dir):
    # 1. Read jsonl
    df = pd.read_json("MATH_hard_train_proxy_tuning_cleaned.jsonl", lines=True)
    dev_df = df[df['split'] == 'dev'].reset_index(drop=True)

    # 2. Convert Dataset
    dev_dataset = Dataset.from_pandas(dev_df)
    dev_dataset = dev_dataset.map(apply_chat_template)

    # 3. Inference
    input_list = list(dev_dataset["input_text_only"])
    sampling_params = SamplingParams(max_tokens=16384, temperature=0.6, skip_special_tokens=False)
    llm = LLM(model=model_dir)
    outputs = llm.generate(input_list, sampling_params)
    outputs = [o.outputs[0].text for o in outputs]
    pred_answers = [extract_boxed_answer(p) for p in outputs]
    label_answers = list(dev_dataset["answer"])

    matches = [is_equiv(p, l) and p is not None for p, l in zip(pred_answers, label_answers)]
    acc = sum(matches) / len(matches) if matches else 0.0

    # 4. Save results
    result_path = os.path.join(model_dir, "eval_results.jsonl")
    with open(result_path, "w", encoding="utf-8") as f:
        for inp, out, pred, label in zip(input_list, outputs, pred_answers, label_answers):
            json.dump({
                "input": inp,
                "model_output": out,
                "extracted_answer": pred,
                "label": label,
                "correct": is_equiv(pred, label) and pred is not None
            }, f, ensure_ascii=False)
            f.write("\n")

    # 5. Rename directory
    parent_dir = os.path.dirname(model_dir)
    basename = os.path.basename(model_dir)
    new_basename = f"{basename}-acc-{acc:.4f}"
    new_model_dir = os.path.join(parent_dir, new_basename)
    os.rename(model_dir, new_model_dir)
    print(f"Renamed model dir to: {new_model_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, help="Path to the model checkpoint directory")
    args = parser.parse_args()
    main(args.model_dir)