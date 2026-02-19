import argparse
import re
import time
import os
import numpy as np
import pandas as pd 
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from math_equivalence import is_equiv
from pathlib import Path

MODEL = "l3lab/L1-Qwen3-8B-Max"
OUTPUT_PATH = Path(__file__).parent / "results.parquet"
def build_prompt(problem: str, target_think_tokens: int) -> str:
    return f"{problem} Think for {target_think_tokens} tokens. <think>"

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

    if exp_val is None:
        exp_val = expected_answer.strip()

    if exp_val is None or gen_val is None:
        print("Expected value is none")
        return False
    
    return is_equiv(gen_val, exp_val)

def main():
    df = pd.read_parquet(Path(__file__).parent / "dataset_splitting" / "test.parquet")#"./data/test_all.parquet")
    targets = np.load(Path(__file__).parent / "regressor" / "test_target_tokens.npy", allow_pickle=True).item()#"./data/test_target_tokens.npy")
    
    df =  df.groupby("question_id", as_index=False).first()
    mask = np.isin(df['question_id'], targets['ids'])
    df = df[mask]
    if len(df) != len(targets['ids']):
        raise ValueError(f"Data length mismatch: DataFrame has {len(df)} rows, targets has {len(targets['ids'])} elements.")
    
    targets = targets['target']

    prompts = []
    for i in range(len(df)):
        prompts.append(build_prompt(df.iloc[i]["prompt"], targets[i]))


    #os.makedirs(args.generated_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    
    llm = LLM(
        model=MODEL,
        dtype="float16", 
        trust_remote_code=True,
        tensor_parallel_size=1, 
        max_model_len=32768,    
    )

    sampling_params = SamplingParams(max_tokens=6000, temperature=0, skip_special_tokens=False)

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
        is_ok = evaluate_answer(row["solution_col"], full_text)

        records.append({
            "question_id": row['question_id'],
            "prompt": prompts[i],
            "solution": row["solution_col"],
            "generated_think_text": think_texts[i],
            "generated_text": full_text,
            "target_think_tokens": int(targets[i]),
            "generated_think_tokens": think_lengths[i],
            "latency_sec": (t1 - t0) / len(prompts),
            "is_correct": bool(is_ok),
        })

    final_df = pd.DataFrame(records)
    out_path = os.path.join(OUTPUT_PATH)
    final_df.to_parquet(out_path, index=False)

if __name__ == "__main__":
    main()