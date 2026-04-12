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

    prompts = [
        format_prompt(
            build_prompt_content(row["problem"], row["dataset"], row["target_think_tokens"], args.method),
            tokenizer,
        )
        for _, row in expanded_df.iterrows()
    ]

    print(f"Starting generation for {len(prompts)} prompts...")
    t0 = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    t1 = time.perf_counter()
    
    records = []
    for i, output in enumerate(outputs):
        row = expanded_df.iloc[i]
        text = output.outputs[0].text
        token_count = len(output.outputs[0].token_ids)

        is_ok, exp_val, gen_val = evaluate_answer(row["solution"], text)

        records.append({
            "unique_id": row["unique_id"],
            "prompt": prompts[i],
            "solution": row["solution"],
            "generated": text,
            "expected_value": exp_val,
            "generated_value": gen_val,
            "token_count": token_count,
            "is_correct": bool(is_ok),
            "latency_sec": (t1 - t0) / len(prompts),
        })

    print("Restoring original dataset order...")
    final_df = pd.DataFrame(records)
    final_df = final_df.sort_values(by="unique_id", ascending=True)

    out_path = os.path.join(args.generated_dir, f"generated_train_data.parquet")
    final_df.to_parquet(out_path, index=False)
    print(f"Saved {len(final_df)} records to {out_path}")

if __name__ == "__main__":
    main()