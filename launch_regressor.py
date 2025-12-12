import argparse
import re
import time
import os
import numpy as np
import pandas as pd 
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from math_equivalence import is_equiv

def build_prompt(problem: str, target_think_tokens: int) -> str:
    return f"{problem} Let’s think step by step inside and output the final answer within boxed{{}}. Think for {target_think_tokens} tokens. <think>"

def extract_boxed(s: str) -> str | None:
    if not s: return None
    m = re.search(r"\\boxed\{([^}]*)\}", s)
    return m.group(1).strip() if m else None

def extract_think_text(full_text: str) -> str:
    match = re.search(r"<think>(.*?)</think>", full_text, flags=re.DOTALL)
    return match.group(1).strip() if match else ""

def evaluate_answer(expected_answer: str, generated_answer: str) -> bool:
    exp_val = extract_boxed(expected_answer)
    gen_val = extract_boxed(generated_answer)
    if exp_val is None or gen_val is None:
        return False
    return is_equiv(gen_val, exp_val)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--generated-dir", required=True)
    parser.add_argument("--file-name", default="regressor_results")
    args = parser.parse_args()

    df = pd.read_parquet("./data/test_all.parquet")
    targets = np.load("./data/test_target_tokens.npy")
    
    if len(df) != len(targets):
        raise ValueError(f"Data length mismatch: DataFrame has {len(df)} rows, targets has {len(targets)} elements.")

    os.makedirs(args.generated_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    
    llm = LLM(
        model=args.model_path,
        dtype="float16", 
        trust_remote_code=True,
        tensor_parallel_size=1, 
        max_model_len=32768,    
    )

    sampling_params = SamplingParams(max_tokens=6000, temperature=0, skip_special_tokens=False)

    prompts = []
    for i in range(len(df)):
        prompts.append(build_prompt(df.iloc[i]["problem"], targets[i]))

    t0 = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    t1 = time.perf_counter()
    
    generated_texts = []
    think_texts = []
    
    for i, output in enumerate(outputs):
        prompt_text = prompts[i]
        generated_suffix = output.outputs[0].text
        full_text = prompt_text + generated_suffix
        generated_texts.append(full_text)
        think_texts.append(extract_think_text(full_text))

    think_encodings = tokenizer(think_texts, add_special_tokens=False)["input_ids"]
    think_lengths = [len(ids) for ids in think_encodings]

    records = []
    for i in range(len(df)):
        row = df.iloc[i]
        full_text = generated_texts[i]
        is_ok = evaluate_answer(row["solution"], full_text)

        records.append({
            "question_id": i,
            "prompt": prompts[i],
            "solution_col": row["solution"],
            "generated_think_text": think_texts[i],
            "generated_text": full_text,
            "target_think_tokens": int(targets[i]),
            "generated_think_tokens": think_lengths[i],
            "latency_sec": (t1 - t0) / len(prompts),
            "is_correct": bool(is_ok),
        })

    final_df = pd.DataFrame(records)
    out_path = os.path.join(args.generated_dir, f"{args.file_name}.parquet")
    final_df.to_parquet(out_path, index=False)

if __name__ == "__main__":
    main()