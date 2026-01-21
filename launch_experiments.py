import argparse, re, time, os
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
    if not m:
        m = re.search(r"\$([^$]*)\$", s)    
    return m.group(1).strip() if m else None

def extract_think_text(full_text: str) -> str:
    match = re.search(r"<think>(.*?)</think>", full_text, flags=re.DOTALL)
    return match.group(1).strip() if match else ""

def evaluate_answer(expected_answer: str, generated_answer: str) -> bool:
    exp_val = extract_boxed(expected_answer)
    gen_val = extract_boxed(generated_answer)
    if exp_val is None or gen_val is None:
        return False, exp_val, gen_val
    return is_equiv(gen_val, exp_val), exp_val, gen_val

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--generated-dir", required=True)
    parser.add_argument("--file-name", required=True)
    args = parser.parse_args()

    print(f"Loading data from {args.data}...")
    df = pd.read_parquet(args.data)
    targets = np.linspace(start=100, stop=5000, num=20, endpoint=True, dtype=int)
    #df = df.sample(n=1, random_state=42).reset_index(drop=True)
    expanded = []
    for _, row in df.iterrows():
        for tgt in targets:
            expanded.append({
                "id": row["id"],
                "dataset": str(row["dataset"]),
                "level": str(row["level"]),
                "problem": str(row["problem"]),
                "solution": str(row["solution"]),
                "target_think_tokens": int(tgt),
            })
    
    expanded_df = pd.DataFrame(expanded)    
    expanded_df = expanded_df.sort_values(by="target_think_tokens", ascending=True)

    os.makedirs(args.generated_dir, exist_ok=True)
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    
    print(f"Loading vLLM model: {args.model_path}")
    llm = LLM(
        model=args.model_path,
        dtype="bfloat16", #On V100, use float16
        trust_remote_code=True,
        tensor_parallel_size=1, 
        max_model_len=32768,    
        #gpu_memory_utilization=0.8,
    )

    sampling_params = SamplingParams(max_tokens=6000, temperature=0, skip_special_tokens=False)

    prompts = [
        build_prompt(row["problem"], row["target_think_tokens"]) 
        for _, row in expanded_df.iterrows()
    ]

    print(f"Starting generation for {len(prompts)} prompts...")
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
    for i in range(len(expanded_df)):
        row = expanded_df.iloc[i]
        full_text = generated_texts[i]
        is_ok, exp_val, gen_val = evaluate_answer(row["solution"], full_text)

        records.append({
            "id": row["id"],
            "dataset": row["dataset"],
            "level": row["level"],
            "prompt": prompts[i],
            "solution_col": row["solution"],
            "extracted_solution": exp_val,
            "extracted_generated_solution": gen_val,
            "is_correct": bool(is_ok),
            "generated_think_text": think_texts[i],
            "generated_text": full_text,
            "target_think_tokens": row["target_think_tokens"],
            "generated_think_tokens": think_lengths[i],
            "latency_sec": (t1 - t0) / len(prompts)
        })

    print("Restoring original dataset order...")
    final_df = pd.DataFrame(records)
    final_df = final_df.sort_values(by="id", ascending=True)

    out_path = os.path.join(args.generated_dir, f"{args.file_name}.parquet")
    final_df.to_parquet(out_path, index=False)
    print(f"Saved {len(final_df)} records to {out_path}")

if __name__ == "__main__":
    main()