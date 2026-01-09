"""
Evaluation Pipeline for Math Reasoning Models

Evaluates models on GSM8K, Olympiad, and AMC datasets.
Reports accuracy and token counts per dataset.
"""

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

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from math_equivalence import is_equiv


# =============================================================================
# Data Classes
# =============================================================================

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


# =============================================================================
# Answer Extraction & Evaluation
# =============================================================================

def extract_boxed(s: str) -> str | None:
    """Extract answer from \\boxed{} notation."""
    if not s:
        return None
    # Handle nested braces by finding the matching closing brace
    match = re.search(r"\\boxed\{", s)
    if not match:
        return None
    
    start = match.end()
    depth = 1
    i = start
    while i < len(s) and depth > 0:
        if s[i] == '{':
            depth += 1
        elif s[i] == '}':
            depth -= 1
        i += 1
    
    if depth == 0:
        return s[start:i-1].strip()
    return None


def extract_numeric_answer(s: str) -> str | None:
    """Extract numeric answer from text (fallback for GSM8K format)."""
    if not s:
        return None
    # Look for #### pattern (GSM8K format)
    match = re.search(r"####\s*(.+?)(?:\n|$)", s)
    if match:
        return match.group(1).strip()
    return None


def evaluate_answer(expected: str, generated: str, dataset_type: str = "math") -> bool:
    """
    Compare expected and generated answers.
    
    Args:
        expected: Ground truth answer
        generated: Model generated answer
        dataset_type: Type of dataset for answer extraction strategy
    """
    # Extract generated answer from boxed notation
    gen_val = extract_boxed(generated)
    
    # Extract expected answer based on dataset type
    if dataset_type == "gsm8k":
        exp_val = extract_numeric_answer(expected)
        if exp_val is None:
            exp_val = expected.strip()
        # For GSM8K, if no boxed answer, try to find numeric answer
        if gen_val is None:
            gen_val = extract_numeric_answer(generated)
    else:
        exp_val = extract_boxed(expected)
        if exp_val is None:
            exp_val = str(expected).strip()
    
    if exp_val is None or gen_val is None:
        return False
    
    return is_equiv(gen_val, exp_val)


# =============================================================================
# Dataset Loaders
# =============================================================================

def load_gsm8k(split: str = "test") -> pd.DataFrame:
    """
    Load GSM8K dataset from HuggingFace.
    
    GSM8K contains grade school math word problems.
    Columns: question, answer (with #### marker for final answer)
    """
    from datasets import load_dataset
    
    dataset = load_dataset("openai/gsm8k", "main", split=split)
    df = pd.DataFrame(dataset)
    
    # Standardize column names
    df = df.rename(columns={
        "question": "problem",
        "answer": "solution"
    })
    df["unique_id"] = [f"gsm8k_{i}" for i in range(len(df))]
    
    return df[["unique_id", "problem", "solution"]]


def load_olympiad(split: str = "test") -> pd.DataFrame:
    """
    Load OlympiadBench dataset from HuggingFace.
    
    Contains Olympiad-level competition math problems.
    """
    from datasets import load_dataset
    
    try:
        # Try loading OlympiadBench
        dataset = load_dataset("lmms-lab/OlympiadBench", split=split)
        df = pd.DataFrame(dataset)
        
        # Standardize column names based on actual dataset structure
        if "question" in df.columns:
            df = df.rename(columns={"question": "problem"})
        if "answer" in df.columns:
            df = df.rename(columns={"answer": "solution"})
        
        df["unique_id"] = [f"olympiad_{i}" for i in range(len(df))]
        
    except Exception as e:
        print(f"Warning: Could not load OlympiadBench directly: {e}")
        print("Attempting to load from alternative source...")
        
        # Fallback: Try MATH dataset Level 5 problems as proxy for Olympiad-level
        try:
            dataset = load_dataset("hendrycks/competition_math", split="test")
            df = pd.DataFrame(dataset)
            # Filter for highest difficulty levels
            if "level" in df.columns:
                df = df[df["level"].isin(["Level 5", "Level 4"])]
            
            df = df.rename(columns={
                "problem": "problem",
                "solution": "solution"
            })
            df["unique_id"] = [f"olympiad_{i}" for i in range(len(df))]
        except Exception as e2:
            print(f"Warning: Could not load competition_math either: {e2}")
            # Return empty DataFrame if both fail
            return pd.DataFrame(columns=["unique_id", "problem", "solution"])
    
    return df[["unique_id", "problem", "solution"]]


def load_amc() -> pd.DataFrame:
    """
    Load AMC (American Mathematics Competition) dataset.
    
    Uses AIME dataset which contains AMC/AIME level problems.
    """
    from datasets import load_dataset
    
    try:
        # Try loading AMC-specific dataset
        df = pd.read_csv("hf://datasets/di-zhang-fdu/AIME_1983_2024/AIME_Dataset_1983_2024.csv")
        
        # Standardize column names
        df = df.rename(columns={
            "Question": "problem",
            "Answer": "solution"
        })
        
        if "ID" in df.columns:
            df["unique_id"] = df["ID"].apply(lambda x: f"amc_{x}")
        else:
            df["unique_id"] = [f"amc_{i}" for i in range(len(df))]
            
    except Exception as e:
        print(f"Warning: Could not load AIME dataset: {e}")
        print("Attempting alternative AMC source...")
        
        try:
            # Try alternative source
            dataset = load_dataset("AI-MO/aimo-validation-amc", split="train")
            df = pd.DataFrame(dataset)
            
            if "problem" not in df.columns and "question" in df.columns:
                df = df.rename(columns={"question": "problem"})
            if "solution" not in df.columns and "answer" in df.columns:
                df = df.rename(columns={"answer": "solution"})
                
            df["unique_id"] = [f"amc_{i}" for i in range(len(df))]
        except Exception as e2:
            print(f"Warning: Could not load alternative AMC source: {e2}")
            return pd.DataFrame(columns=["unique_id", "problem", "solution"])
    
    return df[["unique_id", "problem", "solution"]]


def load_dataset_by_name(
    name: Literal["gsm8k", "olympiad", "amc"],
    split: str = "test",
    sample_size: int | None = None
) -> pd.DataFrame:
    """
    Load a dataset by name.
    
    Args:
        name: Dataset name (gsm8k, olympiad, or amc)
        split: Data split to use
        sample_size: Optional number of samples to use (for testing)
    
    Returns:
        DataFrame with columns: unique_id, problem, solution
    """
    loaders = {
        "gsm8k": lambda: load_gsm8k(split),
        "olympiad": lambda: load_olympiad(split),
        "amc": load_amc,
    }
    
    if name not in loaders:
        raise ValueError(f"Unknown dataset: {name}. Choose from: {list(loaders.keys())}")
    
    df = loaders[name]()
    
    if sample_size is not None and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    
    print(f"Loaded {name}: {len(df)} problems")
    return df


# =============================================================================
# Prompt Creation
# =============================================================================

def create_eval_prompt(problem: str, use_cot: bool = True) -> str:
    """
    Create evaluation prompt for a math problem.
    
    Args:
        problem: The math problem text
        use_cot: Whether to use chain-of-thought prompting
    """
    if use_cot:
        return f"{problem}\n\nLet's think step by step and output the final answer within \\boxed{{}}."
    else:
        return f"{problem}\n\nOutput the final answer within \\boxed{{}}."


def create_prompts_batch(
    df: pd.DataFrame,
    tokenizer,
    use_cot: bool = True,
    enable_thinking: bool = False
) -> list[str]:
    """
    Create prompts for a batch of problems.
    
    Args:
        df: DataFrame with 'problem' column
        tokenizer: Model tokenizer
        use_cot: Whether to use chain-of-thought
        enable_thinking: Whether to enable thinking mode (for Qwen3 etc.)
    
    Returns:
        List of formatted prompts
    """
    prompts = []
    
    for _, row in df.iterrows():
        content = create_eval_prompt(row["problem"], use_cot)
        messages = [{"role": "user", "content": content}]
        
        # Try to apply chat template with thinking enabled
        try:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking
            )
        except TypeError:
            # Fallback if enable_thinking not supported
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        
        prompts.append(prompt)
    
    return prompts


# =============================================================================
# Model Evaluation
# =============================================================================

def evaluate_dataset(
    llm: LLM,
    tokenizer,
    df: pd.DataFrame,
    dataset_name: str,
    sampling_params: SamplingParams,
    use_cot: bool = True,
    enable_thinking: bool = False
) -> EvalResult:
    """
    Evaluate a model on a single dataset.
    
    Args:
        llm: vLLM model instance
        tokenizer: Model tokenizer
        df: Dataset DataFrame
        dataset_name: Name of the dataset
        sampling_params: vLLM sampling parameters
        use_cot: Use chain-of-thought prompting
        enable_thinking: Enable thinking mode
    
    Returns:
        EvalResult with accuracy and token counts
    """
    print(f"\n{'='*60}")
    print(f"Evaluating on {dataset_name} ({len(df)} problems)")
    print(f"{'='*60}")
    
    # Create prompts
    prompts = create_prompts_batch(df, tokenizer, use_cot, enable_thinking)
    
    # Generate responses
    outputs = llm.generate(prompts, sampling_params)
    
    # Process results
    results = []
    total_tokens = 0
    num_correct = 0
    
    dataset_type = "gsm8k" if dataset_name == "gsm8k" else "math"
    
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        
        # Count tokens
        token_count = len(tokenizer(generated_text, add_special_tokens=False)["input_ids"])
        total_tokens += token_count
        
        # Evaluate correctness
        is_correct = evaluate_answer(
            df.iloc[i]["solution"],
            generated_text,
            dataset_type
        )
        if is_correct:
            num_correct += 1
        
        results.append({
            "unique_id": df.iloc[i]["unique_id"],
            "problem": df.iloc[i]["problem"],
            "solution": df.iloc[i]["solution"],
            "generated": generated_text,
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
    datasets: list[str] = ["gsm8k", "olympiad", "amc"],
    output_dir: str = "./eval_results",
    max_tokens: int = 4096,
    temperature: float = 0.0,
    sample_size: int | None = None,
    use_cot: bool = True,
    enable_thinking: bool = False,
    dtype: str = "bfloat16",
    max_model_len: int = 8192,
) -> dict[str, EvalResult]:
    """
    Run the full evaluation pipeline on multiple datasets.
    
    Args:
        model_path: Path to model or HuggingFace model ID
        datasets: List of dataset names to evaluate
        output_dir: Directory to save results
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0 for greedy)
        sample_size: Optional sample size per dataset
        use_cot: Use chain-of-thought prompting
        enable_thinking: Enable thinking mode
        dtype: Model data type
        max_model_len: Maximum model context length
    
    Returns:
        Dictionary mapping dataset names to EvalResults
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'#'*60}")
    print(f"# Evaluation Pipeline")
    print(f"# Model: {model_path}")
    print(f"# Datasets: {datasets}")
    print(f"{'#'*60}\n")
    
    # Initialize model
    print("Loading model...")
    llm = LLM(
        model=model_path,
        dtype=dtype,
        max_model_len=max_model_len,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
    )
    
    # Evaluate each dataset
    results: dict[str, EvalResult] = {}
    
    for dataset_name in datasets:
        try:
            # Load dataset
            df = load_dataset_by_name(dataset_name, sample_size=sample_size)
            
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
                use_cot=use_cot,
                enable_thinking=enable_thinking
            )
            
            results[dataset_name] = result
            
            # Save individual results
            result_file = output_path / f"{dataset_name}_results.parquet"
            result.results_df.to_parquet(result_file)
            print(f"Saved results to {result_file}")
            
        except Exception as e:
            print(f"Error evaluating {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
    
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
    print(f"\n{'='*70}")
    print(f"{'EVALUATION SUMMARY':^70}")
    print(f"{'='*70}")
    print(f"{'Dataset':<15} {'Accuracy':>12} {'Correct':>10} {'Total':>8} {'Avg Tokens':>12}")
    print(f"{'-'*70}")
    
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
    summary_file = output_path / "evaluation_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"Summary saved to {summary_file}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate math reasoning models on GSM8K, Olympiad, and AMC datasets"
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
        default=["gsm8k", "olympiad", "amc"],
        choices=["gsm8k", "olympiad", "amc"],
        help="Datasets to evaluate on"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./eval_results",
        help="Directory to save results"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0 for greedy)"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Sample size per dataset (for testing)"
    )
    parser.add_argument(
        "--no-cot",
        action="store_true",
        help="Disable chain-of-thought prompting"
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable thinking mode (for Qwen3 etc.)"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Model data type"
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=8192,
        help="Maximum model context length"
    )
    
    args = parser.parse_args()
    
    run_evaluation_pipeline(
        model_path=args.model_path,
        datasets=args.datasets,
        output_dir=args.output_dir,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        sample_size=args.sample_size,
        use_cot=not args.no_cot,
        enable_thinking=args.enable_thinking,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
    )


if __name__ == "__main__":
    main()

