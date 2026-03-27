import argparse
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
sys.path.insert(0, str(Path(__file__).parent.parent))
from data.processing.math_equivalence import is_equiv

EVAL_DATASET_PATH = Path(__file__).parent.parent / "data" / "raw" / "eval_data.parquet"
AVAILABLE_DATASETS = ["math-500", "gsm8k", "olympiad", "amc", "aime-250"]
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
    
    # Apply sample size per dataset
    if sample_size is not None:
        df = df.groupby("dataset").apply(
            lambda x: x.sample(n=min(sample_size, len(x)), random_state=42)
        ).reset_index(drop=True)
    
    # Add unique_id column for compatibility
    df["unique_id"] = df["id"].astype(str) + "-" + df["dataset"]
    
    print(f"Loaded {len(df)} problems from: {df['dataset'].value_counts().to_dict()}")
    return df

def extract_boxed(s: str) -> str | None:
    if not s:
        return None
    # MATH & AIME — take LAST boxed match (final answer, not thinking)
    matches = re.findall(r"\\{1,2}boxed\{([^}]*)\}", s)
    if matches:
        return matches[-1].strip()
    # GSM8K
    matches = re.findall(r"(?m)^[ \t]*####[ \t]*([^\n\r#]+?)[ \t]*$", s)
    if matches:
        return matches[-1].strip()
    # Olympiad
    matches = re.findall(r"\$([^$]*)\$", s)
    if matches:
        return matches[-1].strip()
    # AMC
    matches = re.findall(r"(?m)^[ \t]*([+-]?\d+(?:\.\d+)?)[ \t]*$", s)
    if matches:
        return matches[-1].strip()
    return s

def evaluate_answer(expected_answer: str, generated_answer: str) -> bool:
    exp_val = extract_boxed(expected_answer)
    gen_val = extract_boxed(generated_answer)
    if exp_val is None or gen_val is None:
        return False, exp_val, gen_val
    return is_equiv(gen_val, exp_val), exp_val, gen_val

def build_prompt(problem: str, dataset_name: str) -> str:
    if dataset_name == "aime-250":
        return f"{problem}"
    return f"{problem}\nLet's think step by step and output the final answer within boxed{{}}."

def evaluate_dataset(
    llm: LLM,
    tokenizer,
    df: pd.DataFrame,
    dataset_name: str,
    sampling_params: SamplingParams,
) -> EvalResult:
    print(f"Evaluating on {dataset_name} ({len(df)} problems)")
        
    # Create prompts
    prompts = [
       build_prompt(row["problem"], dataset_name) 
       for _, row in df.iterrows()
    ]    
    #prompts = df["problem"].tolist()
    # Generate responses
    outputs = llm.generate(prompts, sampling_params)
    
    # Process results
    results = []
    total_tokens = 0
    num_correct = 0
        
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        
        # Count tokens
        token_count = len(output.outputs[0].token_ids)
        total_tokens += token_count
        
        # Evaluate correctness
        is_correct, expected_value, generated_value = evaluate_answer(
            str(df.iloc[i]["solution"]),
            generated_text,
        )
        if is_correct:
            num_correct += 1
        
        results.append({
            "unique_id": df.iloc[i]["unique_id"],
            "prompt": prompts[i],
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
) -> dict[str, EvalResult]:
    output_path = Path(output_dir) / Path(model_path).stem
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Evaluation Pipeline")
    print(f"Model: {model_path}")
    print(f"Datasets: {datasets}")
    
    # Initialize model
    print("Loading model...")
    llm = LLM(
        model=model_path,
        dtype="bfloat16", #On V100, use float16
        trust_remote_code=True,
        tensor_parallel_size=1, 
        max_model_len=32000,    
        #gpu_memory_utilization=0.8,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    sampling_params = SamplingParams(max_tokens=32000, temperature=0, skip_special_tokens=False)
    # Evaluate each dataset
    results: dict[str, EvalResult] = {}
    
    for dataset_name in datasets:
        df =load_eval_dataset(datasets=[dataset_name])
        if len(df) == 0:
            print(f"Warning: {dataset_name} is empty, skipping...")
            continue
        
        # Evaluate
        result = evaluate_dataset(
            llm=llm,
            tokenizer=tokenizer,
            df=df,
            dataset_name=dataset_name,
            sampling_params=sampling_params,
        )
        
        results[dataset_name] = result
        
        # Save individual results
        result_file = output_path / f"{dataset_name}_results.parquet"
        result.results_df.to_parquet(result_file)
        print(f"Saved results to {result_file}")
            
    # Clean up
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    
    # Print summary
    print_summary(results)
    
    # Save summary
    save_summary(results, output_path)
    
    return results


def print_summary(results: dict[str, EvalResult]):
    """Print evaluation summary to console."""
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


def save_summary(results: dict[str, EvalResult], output_path: Path):
    """Save evaluation summary to CSV."""
    summary_data = []
    
    for name, result in results.items():
        summary_data.append({
            "dataset": name,
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
        description="Evaluate math reasoning models on math-500, gsm8k, and olympiad datasets"
    )
    
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to model or HuggingFace model ID"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["math-500", "gsm8k", "olympiad"],
        choices=["math-500", "gsm8k", "olympiad", "all", "aime-250"],
        help="Datasets to evaluate on (math-500, gsm8k, olympiad, or 'all')"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./eval_results",
        help="Directory to save results"
    )
    
    args = parser.parse_args()
    
    # Handle 'all' option
    datasets = args.datasets
    if "all" in datasets:
        datasets = AVAILABLE_DATASETS
    
    run_evaluation_pipeline(
        model_path=args.model_path,
        datasets=datasets,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()

