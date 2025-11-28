#!/usr/bin/env python3
from __future__ import annotations
import argparse, re, time, os
import numpy as np
import pandas as pd 
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from math_equivalence import is_equiv

# --- Helper Functions ---
def build_prompt(problem: str, target_think_tokens: int) -> str:
    return f"{problem} Let’s think step by step inside and output the final answer within boxed{{}}. Think for {target_think_tokens} tokens. <think>"

def extract_boxed(s: str):
    if not s: return None
    m = re.search(r"\\boxed\{([^}]*)\}", s)
    return m.group(1).strip() if m else None

def evaluate_answer(expected_answer, generated_answer):
    exp_val = extract_boxed(expected_answer)
    gen_val = extract_boxed(generated_answer)
    if exp_val is None or gen_val is None:
        return False
    return is_equiv(gen_val, exp_val)

def extract_think_text(full_text: str):
    match = re.search(r"<think>(.*?)</think>", full_text, flags=re.DOTALL)
    return match.group(1).strip() if match else ""

# --- Main ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--generated-dir", required=True)
    parser.add_argument("--save-every", type=int, default=500, help="Save a CSV every N prompts")
    args = parser.parse_args()

    # 1. Load Data
    print(f"Loading data from {args.data}...")
    df = pd.read_parquet(args.data)
    targets = np.linspace(start=100, stop=2500, num=10, endpoint=True, dtype=int)

    # 2. Expand Data
    print("Expanding dataset...")
    expanded = []
    for qid, row in df.iterrows():
        for tgt in targets:
            expanded.append({
                "question_id": qid,
                "problem": str(row["problem"]),
                "solution": str(row["solution"]),
                "target_think_tokens": int(tgt),
            })
    expanded_df = pd.DataFrame(expanded)
    
    # 3. Setup Resources
    os.makedirs(args.generated_dir, exist_ok=True)
    
    # Load tokenizer for counting (CPU)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    
    # Load vLLM (GPU)
    print(f"Loading vLLM model: {args.model_path}")
    llm = LLM(
        model=args.model_path,
        dtype="bfloat16",
        trust_remote_code=True,
        tensor_parallel_size=1, 
        max_model_len=32768,    
    )

    sampling_params = SamplingParams(max_tokens=4096, temperature=0)

    # 4. Process in Chunks (So we save progress)
    total_rows = len(expanded_df)
    chunk_size = args.save_every
    
    print(f"Processing {total_rows} prompts in chunks of {chunk_size}...")

    for start_idx in range(0, total_rows, chunk_size):
        end_idx = min(start_idx + chunk_size, total_rows)
        batch_id = (start_idx // chunk_size) + 1
        
        # Slice the dataframe
        chunk_df = expanded_df.iloc[start_idx:end_idx]
        
        # Build prompts for this chunk
        chunk_prompts = [
            build_prompt(row["problem"], row["target_think_tokens"]) 
            for _, row in chunk_df.iterrows()
        ]

        print(f"Generating Chunk {batch_id} ({start_idx} to {end_idx})...")
        t0 = time.perf_counter()
        
        # vLLM handles the internal batching for this chunk
        outputs = llm.generate(chunk_prompts, sampling_params)
        
        t1 = time.perf_counter()

        # Process Results for this chunk
        generated_texts = []
        think_texts = []
        
        for i, output in enumerate(outputs):
            prompt_text = chunk_prompts[i]
            generated_suffix = output.outputs[0].text
            full_text = prompt_text + generated_suffix
            generated_texts.append(full_text)
            think_texts.append(extract_think_text(full_text))

        # Vectorized Token Counting
        think_encodings = tokenizer(think_texts, add_special_tokens=False)["input_ids"]
        think_lengths = [len(ids) for ids in think_encodings]

        # Build Records
        records = []
        for i in range(len(chunk_df)):
            row = chunk_df.iloc[i]
            full_text = generated_texts[i]
            is_ok = evaluate_answer(row["solution"], full_text)

            records.append({
                "question_id": row["question_id"],
                "prompt": chunk_prompts[i],
                "solution_col": row["solution"],
                "generated_think_text": think_texts[i],
                "generated_text": full_text,
                "target_think_tokens": row["target_think_tokens"],
                "generated_think_tokens": think_lengths[i],
                "latency_sec": (t1 - t0) / len(chunk_prompts),
                "is_correct": bool(is_ok),
            })

        # SAVE IMMEDIATELY
        out_path = os.path.join(args.generated_dir, f"generated_chunk_{batch_id}.csv")
        pd.DataFrame(records).to_csv(out_path, index=False)
        print(f"Saved {out_path}")

    print("All chunks processed.")

if __name__ == "__main__":
    main()