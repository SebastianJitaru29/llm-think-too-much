import argparse
import json
import re
import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd
import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import numpy as np

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from data.processing.math_equivalence import is_equiv

EVAL_DATASET_PATH = Path(__file__).parent.parent / "data" / "raw" / "eval_data.parquet"
AVAILABLE_DATASETS = ["math-500", "gsm8k", "olympiad", "amc", "aime-250"]

# ---------------------------------------------------------------------------
# NOWAIT keyword suppression
# ---------------------------------------------------------------------------
# Full list from the paper (Wang et al., EMNLP 2025 Findings)
NOWAIT_KEYWORDS_FULL = [
    "wait", "alternatively", "hmm", "but", "however",
    "alternative", "another", "check", "double-check",
    "oh", "maybe", "verify", "other", "again", "now", "ah", "any",
]

# Conservative subset — safer for small (≤ 3B) models where "but", "now",
# "any", "other" are common connectives, not reflection triggers
NOWAIT_KEYWORDS_SAFE = [
    "wait", "alternatively", "hmm", "however",
    "double-check", "maybe", "ah", "oh",
]


def build_nowait_suppress_ids(
    tokenizer,
    keywords: list[str] | None = None,
    match: str = "exact",  # "exact" or "prefix"
) -> set[int]:
    """Scan the full vocabulary and collect token IDs to suppress.
    
    Args:
        tokenizer: HuggingFace tokenizer.
        keywords: list of lowercase keywords. Defaults to safe subset.
        match: 
            "exact"  — suppress only tokens whose decoded text IS the keyword
                       (with optional leading space / punctuation).  Safest.
            "prefix" — suppress any token that STARTS WITH the keyword.
                       Closer to the original paper but riskier on small models.
    """
    if keywords is None:
        keywords = NOWAIT_KEYWORDS_SAFE

    suppress_ids: set[int] = set()
    vocab_size = tokenizer.vocab_size

    for token_id in range(vocab_size):
        try:
            decoded = tokenizer.decode([token_id])
        except Exception:
            continue

        # Normalize: strip leading whitespace/punctuation, lowercase
        cleaned = decoded.lstrip(" \t\n.,:;!?").lower()

        if match == "exact":
            if cleaned in keywords:
                suppress_ids.add(token_id)
        elif match == "prefix":
            for kw in keywords:
                if cleaned.startswith(kw):
                    suppress_ids.add(token_id)
                    break

    return suppress_ids


def make_nowait_processor(suppress_ids: set[int]):
    """Return a vLLM-compatible logits processor function."""
    suppress_list = list(suppress_ids)

    def nowait_logits_processor(token_ids, logits):
        logits[suppress_list] = float("-inf")
        return logits

    return nowait_logits_processor


# ---------------------------------------------------------------------------
# Core eval code (unchanged logic, NOWAIT wired in via SamplingParams)
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    """Results for a single dataset evaluation."""
    dataset_name: str
    accuracy: float
    num_correct: int
    num_total: int
    avg_tokens: float
    total_tokens: int
    results_df: pd.DataFrame


def load_eval_dataset(
    datasets: list[str] | None = None,
    sample_size: int | None = None,
) -> pd.DataFrame:
    if not EVAL_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found at {EVAL_DATASET_PATH}. "
            "Please ensure the data file exists."
        )
    
    df = pd.read_parquet(EVAL_DATASET_PATH)
    df = df[df["dataset"].isin(datasets)]
    
    if sample_size is not None:
        df = df.groupby("dataset").apply(
            lambda x: x.sample(n=min(sample_size, len(x)), random_state=42)
        ).reset_index(drop=True)
    
    df["unique_id"] = df["id"].astype(str) + "-" + df["dataset"]
    
    print(f"Loaded {len(df)} problems from: {df['dataset'].value_counts().to_dict()}")
    return df


def extract_boxed(s: str) -> str | None:
    if not s:
        return None
    matches = re.findall(r"\\{1,2}boxed\{([^}]*)\}", s)
    if matches:
        return ", ".join(m.strip() for m in matches)
    matches = re.findall(r"(?m)^[ \t]*####[ \t]*([^\n\r#]+?)[ \t]*$", s)
    if matches:
        return ", ".join(m.strip() for m in matches)
    matches = re.findall(r"\$([^$]*)\$", s)
    if matches:
        return ", ".join(m.strip() for m in matches)
    matches = re.findall(r"(?m)^[ \t]*([+-]?\d+(?:\.\d+)?)[ \t]*$", s)
    if matches:
        return ", ".join(m.strip() for m in matches)
    return s


def evaluate_answer(expected_answer: str, generated_answer: str) -> bool:
    exp_val = extract_boxed(expected_answer)
    gen_val = extract_boxed(generated_answer)
    if exp_val is None or gen_val is None:
        return False, exp_val, gen_val
    return is_equiv(gen_val, exp_val), exp_val, gen_val


def build_prompt(problem: str) -> str:
    return f"{problem} Let's think step by step inside and output the final answer within boxed{{}}."


def evaluate_dataset(
    llm: LLM,
    tokenizer,
    df: pd.DataFrame,
    dataset_name: str,
    sampling_params: SamplingParams,
) -> EvalResult:
    print(f"Evaluating on {dataset_name} ({len(df)} problems)")
        
    prompts = [
       build_prompt(row["problem"]) 
       for _, row in df.iterrows()
    ]    
    outputs = llm.generate(prompts, sampling_params)
    
    results = []
    total_tokens = 0
    num_correct = 0
        
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        token_count = len(tokenizer(generated_text, add_special_tokens=False)["input_ids"])
        total_tokens += token_count
        
        is_correct, expected_value, generated_value = evaluate_answer(
            str(df.iloc[i]["solution"]),
            generated_text,
        )
        if is_correct:
            num_correct += 1
        
        results.append({
            "unique_id": df.iloc[i]["unique_id"],
            "problem": df.iloc[i]["problem"],
            "solution": df.iloc[i]["solution"],
            "generated": generated_text,
            "expected_value": expected_value,
            "generated_value": generated_value,
            "token_count": token_count,
            "is_correct": is_correct
        })
    
    results_df = pd.DataFrame(results)
    accuracy = num_correct / len(df) if len(df) > 0 else 0.0
    avg_tokens = total_tokens / len(df) if len(df) > 0 else 0.0
    
    return EvalResult(
        dataset_name=dataset_name,
        accuracy=accuracy,
        num_correct=num_correct,
        num_total=len(df),
        avg_tokens=avg_tokens,
        total_tokens=total_tokens,
        results_df=results_df
    )


def run_evaluation_pipeline(
    model_path: str,
    datasets: list[str] = ["math-500", "gsm8k", "olympiad"],
    output_dir: str = "./eval_results",
    mode: str = "baseline",
    nowait_keywords: str = "safe",
    nowait_match: str = "exact",
    nowait_custom_list: list[str] | None = None,
) -> dict[str, EvalResult]:
    # Tag output directory with mode
    run_name = f"{Path(model_path).stem}_{mode}"
    if mode == "nowait":
        run_name += f"_{nowait_keywords}_{nowait_match}"
    output_path = Path(output_dir) / run_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"{'='*70}")
    print(f"Evaluation Pipeline")
    print(f"  Model:    {model_path}")
    print(f"  Mode:     {mode}")
    print(f"  Datasets: {datasets}")
    if mode == "nowait":
        print(f"  Keywords: {nowait_keywords}")
        print(f"  Match:    {nowait_match}")
    print(f"  Output:   {output_path}")
    print(f"{'='*70}")
    
    # Initialize model
    print("Loading model...")
    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        trust_remote_code=True,
        tensor_parallel_size=1, 
        max_model_len=32768,    
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # Build sampling params — with or without NOWAIT
    logits_processors = []
    
    if mode == "nowait":
        if nowait_custom_list:
            kw_list = nowait_custom_list
        elif nowait_keywords == "full":
            kw_list = NOWAIT_KEYWORDS_FULL
        else:
            kw_list = NOWAIT_KEYWORDS_SAFE
        print(f"Building NOWAIT suppress set from {len(kw_list)} keywords...")
        suppress_ids = build_nowait_suppress_ids(
            tokenizer, keywords=kw_list, match=nowait_match,
        )
        print(f"  Suppressing {len(suppress_ids)} token IDs")
        logits_processors.append(make_nowait_processor(suppress_ids))
        
        # Save the suppressed tokens for reproducibility
        suppressed_tokens = {
            tid: tokenizer.decode([tid]) for tid in sorted(suppress_ids)
        }
        pd.DataFrame(
            list(suppressed_tokens.items()), columns=["token_id", "decoded"]
        ).to_csv(output_path / "suppressed_tokens.csv", index=False)
        print(f"  Saved suppressed token list to {output_path / 'suppressed_tokens.csv'}")

    sampling_params = SamplingParams(
        max_tokens=32000,
        temperature=0,
        skip_special_tokens=False,
        logits_processors=logits_processors if logits_processors else None,
    )
    
    # Evaluate each dataset
    results: dict[str, EvalResult] = {}
    
    for dataset_name in datasets:
        df = load_eval_dataset(datasets=[dataset_name])
        if len(df) == 0:
            print(f"Warning: {dataset_name} is empty, skipping...")
            continue
        
        result = evaluate_dataset(
            llm=llm,
            tokenizer=tokenizer,
            df=df,
            dataset_name=dataset_name,
            sampling_params=sampling_params,
        )
        
        results[dataset_name] = result
        
        result_file = output_path / f"{dataset_name}_results.parquet"
        result.results_df.to_parquet(result_file)
        print(f"Saved results to {result_file}")
            
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    
    print_summary(results, mode=mode)
    save_summary(results, output_path, mode=mode)
    
    return results


def print_summary(results: dict[str, EvalResult], mode: str = "baseline"):
    print(f"\n{'='*70}")
    print(f"  Summary  [{mode.upper()}]")
    print(f"{'='*70}")
    print(f"{'Dataset':<15} {'Accuracy':>12} {'Correct':>10} {'Total':>8} {'Avg Tokens':>12}")
    
    total_correct = 0
    total_problems = 0
    total_tokens = 0
    
    for name, result in results.items():
        print(f"{name:<15} {result.accuracy*100:>11.2f}% {result.num_correct:>10} {result.num_total:>8} {result.avg_tokens:>12.1f}")
        total_correct += result.num_correct
        total_problems += result.num_total
        total_tokens += result.total_tokens
    
    print(f"{'-'*70}")
    if total_problems > 0:
        overall_acc = total_correct / total_problems
        overall_avg_tokens = total_tokens / total_problems
        print(f"{'OVERALL':<15} {overall_acc*100:>11.2f}% {total_correct:>10} {total_problems:>8} {overall_avg_tokens:>12.1f}")
    print(f"{'='*70}\n")


def save_summary(results: dict[str, EvalResult], output_path: Path, mode: str = "baseline"):
    summary_data = []
    
    for name, result in results.items():
        summary_data.append({
            "dataset": name,
            "mode": mode,
            "accuracy": result.accuracy,
            "num_correct": result.num_correct,
            "num_total": result.num_total,
            "avg_tokens": result.avg_tokens,
            "total_tokens": result.total_tokens
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_file = output_path / "evaluation_summary.parquet"
    summary_df.to_parquet(summary_file, index=False)
    print(f"Summary saved to {summary_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate math reasoning models with optional NOWAIT baseline"
    )
    
    parser.add_argument(
        "--model-path", type=str, required=True,
        help="Path to model or HuggingFace model ID"
    )
    parser.add_argument(
        "--datasets", type=str, nargs="+",
        default=["math-500", "gsm8k", "olympiad"],
        choices=["math-500", "gsm8k", "olympiad", "all", "aime-250", "amc"],
        help="Datasets to evaluate on"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./eval_results",
        help="Directory to save results"
    )
    parser.add_argument(
        "--mode", type=str, default="baseline",
        choices=["baseline", "nowait"],
        help="Evaluation mode: baseline (normal) or nowait (suppress reflection tokens)"
    )
    parser.add_argument(
        "--nowait-keywords", type=str, default="safe",
        choices=["safe", "full", "custom"],
        help="NOWAIT keyword set: 'safe' (8 keywords, good for small models), "
             "'full' (17 keywords, from the paper), or 'custom' (from json file)"
    )
    parser.add_argument(
        "--nowait-custom", type=str, default=None,
        help="Path to discovered_keywords.json from nowait_discover_keywords.py. "
             "Used when --nowait-keywords custom"
    )
    parser.add_argument(
        "--nowait-match", type=str, default="exact",
        choices=["exact", "prefix"],
        help="How to match keywords to tokens: "
             "'exact' (decoded text == keyword) or 'prefix' (starts with keyword)"
    )
    
    args = parser.parse_args()
    
    datasets = args.datasets
    if "all" in datasets:
        datasets = AVAILABLE_DATASETS
    
    # Load custom keywords if specified
    custom_keywords = None
    if args.nowait_keywords == "custom":
        if not args.nowait_custom:
            parser.error("--nowait-custom path required when --nowait-keywords custom")
        with open(args.nowait_custom) as f:
            data = json.load(f)
        custom_keywords = data["suggested_suppress"]
        print(f"Loaded {len(custom_keywords)} custom keywords: {custom_keywords}")
    
    run_evaluation_pipeline(
        model_path=args.model_path,
        datasets=datasets,
        output_dir=args.output_dir,
        mode=args.mode,
        nowait_keywords=args.nowait_keywords,
        nowait_match=args.nowait_match,
        nowait_custom_list=custom_keywords,
    )


if __name__ == "__main__":
    main()