import argparse, re, time, os
import numpy as np
import pandas as pd
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from evaluation.evaluation import evaluate_answer
import math

def build_prompt_content(problem: str, dataset_name: str, target_think_tokens: int, method: str = "token") -> str:
    """Build the user-facing problem content with target token/sentence instruction."""
    if dataset_name == "aime-250":
        content = problem
    else:
        content = f"{problem}\nLet’s think step by step and output the final answer within boxed{{}}."
    if method == "sentence":
        content = f"{content} Use less than {math.ceil(target_think_tokens // 80)} sentences."
    else:
        content = f"{content} Think for {target_think_tokens} tokens."
    return content


def format_prompt(content: str, tokenizer):
    """Apply chat template to content."""
    messages = [{"role": "user", "content": content}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--generated-dir", required=True)
    parser.add_argument("--method", type=str, default="token",
                        choices=["token", "sentence"],
                        help="Prompt method: 'token' = Think for N tokens, 'sentence' = Use less than N sentences")
    parser.add_argument("--batch-size", type=int, default=100,
                        help="Number of prompts per batch (intermediate save after each batch)")
    args = parser.parse_args()

    print(f"Loading data from {args.data}...")
    df = pd.read_parquet(args.data)
    targets = np.linspace(start=100, stop=5000, num=20, endpoint=True, dtype=int)
    print(df.shape)
    df = df[:int(df.shape[0]/2)]
    expanded = []
    for _, row in df.iterrows():
        for tgt in targets:
            expanded.append({
                "id": row["id"],
                "unique_id": str(row["id"]) + "-" + str(row["dataset"]),
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

    expanded_df = expanded_df.reset_index(drop=True)

    prompts = [
        format_prompt(
            build_prompt_content(row["problem"], row["dataset"], row["target_think_tokens"], args.method),
            tokenizer,
        )
        for _, row in expanded_df.iterrows()
    ]

    out_path = os.path.join(args.generated_dir, "generated_train_data.parquet")
    partial_path = os.path.join(args.generated_dir, "generated_train_data_partial.parquet")

    # Resume from partial results if they exist
    records = []
    start_idx = 0
    if os.path.exists(partial_path):
        partial_df = pd.read_parquet(partial_path)
        records = partial_df.to_dict("records")
        start_idx = len(records)
        print(f"Resuming from {partial_path} — {start_idx}/{len(prompts)} already done")

    total = len(prompts)
    batch_size = args.batch_size

    for batch_start in range(start_idx, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch_prompts = prompts[batch_start:batch_end]

        print(f"Generating batch {batch_start}–{batch_end} / {total} ...")
        t0 = time.perf_counter()
        outputs = llm.generate(batch_prompts, sampling_params)
        t1 = time.perf_counter()
        batch_latency = (t1 - t0) / len(batch_prompts)

        for j, output in enumerate(outputs):
            row = expanded_df.iloc[batch_start + j]
            text = output.outputs[0].text
            token_count = len(output.outputs[0].token_ids)
            is_ok, exp_val, gen_val = evaluate_answer(row["solution"], text)

            records.append({
                "unique_id": row["unique_id"],
                "prompt": batch_prompts[j],
                "solution": row["solution"],
                "generated": text,
                "expected_value": exp_val,
                "generated_value": gen_val,
                "token_count": token_count,
                "is_correct": bool(is_ok),
                "latency_sec": batch_latency,
            })

        # Save intermediate results
        partial_df = pd.DataFrame(records)
        partial_df.to_parquet(partial_path, index=False)
        print(f"  Saved {len(records)}/{total} records to {partial_path}")

    print("Restoring original dataset order...")
    final_df = pd.DataFrame(records)
    final_df = final_df.sort_values(by="unique_id", ascending=True)

    final_df.to_parquet(out_path, index=False)
    print(f"Saved {len(final_df)} records to {out_path}")

    # Clean up partial file
    if os.path.exists(partial_path):
        os.remove(partial_path)
        print(f"Removed partial file: {partial_path}")

if __name__ == "__main__":
    main()