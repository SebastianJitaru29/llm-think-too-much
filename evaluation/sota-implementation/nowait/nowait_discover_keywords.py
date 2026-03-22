"""
NOWAIT Replication — Stage 1: Discover Reflection Keywords

Runs the base model on a sample of problems, splits CoT by paragraph,
and counts leading words to find the model's reflection triggers.

This replicates the paper's methodology:
  "We conduct 32 independent runs of QwQ-32B on AIME 2025.
   Using '\\n\\n' as delimiters, we identify the 15 most frequent
   monolingual words as our identified keywords."

Since we're on a 1.5B model, the reflection patterns may differ.

Usage:
    python nowait_discover_keywords.py \
        --model-path ./models/DeepSeek-R1-Distill-Qwen-1.5B \
        --datasets math-500 \
        --num-samples 100 \
        --top-k 20
"""

import argparse
import re
import json
from collections import Counter
from pathlib import Path

import pandas as pd
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

import sys
EVAL_DATASET_PATH = Path(__file__).parent.parent.parent.parent / "data" / "raw" / "eval_data.parquet"


def load_problems(datasets: list[str], num_samples: int | None = None) -> list[str]:
    df = pd.read_parquet(EVAL_DATASET_PATH)
    df = df[df["dataset"].isin(datasets)]
    if num_samples and num_samples < len(df):
        df = df.sample(n=num_samples, random_state=42)
    print(f"Loaded {len(df)} problems from {datasets}")
    return df["problem"].tolist()


def build_prompt(problem: str) -> str:
    return f"{problem} Let's think step by step inside and output the final answer within boxed{{}}."


def analyze_cot_blocks(texts: list[str], top_k: int = 20) -> Counter:
    """Split each CoT by \\n\\n and count first words of each block."""
    leading_words = Counter()
    total_blocks = 0

    for text in texts:
        # Split by double newline (paragraph boundaries)
        blocks = re.split(r"\n\n+", text.strip())

        for block in blocks:
            block = block.strip()
            if not block:
                continue
            total_blocks += 1

            # Get first word, strip punctuation
            first_token = re.split(r"\s+", block)[0]
            # Clean: remove leading punctuation, lowercase
            cleaned = re.sub(r"^[^a-zA-Z]+", "", first_token).lower()
            if cleaned and cleaned.isascii() and len(cleaned) > 1:
                leading_words[cleaned] += 1

    print(f"Analyzed {len(texts)} CoTs, {total_blocks} paragraph blocks")
    return leading_words


def main():
    parser = argparse.ArgumentParser(
        description="Discover NOWAIT reflection keywords for a specific model"
    )
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument(
        "--datasets", type=str, nargs="+", default=["math-500"],
    )
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--num-runs", type=int, default=1,
                        help="Number of independent runs per problem (paper uses 32, "
                             "but 1 is fine with temp=0)")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--output-dir", type=str, default="./nowait_analysis")
    args = parser.parse_args()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load problems
    problems = load_problems(args.datasets, args.num_samples)
    prompts = [build_prompt(p) for p in problems]

    # If multiple runs with temperature > 0, duplicate prompts
    if args.num_runs > 1 and args.temperature > 0:
        prompts = prompts * args.num_runs
        print(f"Running {args.num_runs} independent generations → {len(prompts)} total")

    # Generate
    print(f"Loading model: {args.model_path}")
    llm = LLM(
        model=args.model_path,
        dtype="bfloat16",
        trust_remote_code=True,
        tensor_parallel_size=1,
        max_model_len=2048,
    )

    sampling_params = SamplingParams(
        max_tokens=2048,
        temperature=args.temperature,
        skip_special_tokens=False,
    )

    print(f"Generating {len(prompts)} responses...")
    outputs = llm.generate(prompts, sampling_params)
    texts = [o.outputs[0].text for o in outputs]

    # Analyze
    print(f"\nAnalyzing paragraph-leading words...")
    leading_words = analyze_cot_blocks(texts, top_k=args.top_k)

    # Print results
    print(f"\n{'='*60}")
    print(f"  Top {args.top_k} Leading Words (Reflection Candidates)")
    print(f"{'='*60}")
    print(f"{'Rank':>4}  {'Word':<20} {'Count':>6} {'% of blocks':>10}")
    print(f"{'-'*60}")

    total = sum(leading_words.values())
    top_words = leading_words.most_common(args.top_k)
    for rank, (word, count) in enumerate(top_words, 1):
        pct = count / total * 100
        print(f"{rank:>4}  {word:<20} {count:>6} {pct:>9.1f}%")

    # Categorize: which are likely reflection vs normal reasoning
    # Known reflection words from the NOWAIT paper
    known_reflection = {
        "wait", "alternatively", "hmm", "however", "maybe",
        "actually", "oh", "ah", "but", "hold",
    }
    known_reasoning = {
        "the", "so", "since", "let", "we", "first", "next",
        "step", "then", "now", "for", "if", "given", "to",
        "therefore", "thus", "hence", "from", "this", "that",
    }

    print(f"\n{'='*60}")
    print(f"  Classification")
    print(f"{'='*60}")
    reflection_candidates = []
    for word, count in top_words:
        if word in known_reflection:
            label = "🔴 REFLECTION (suppress)"
            reflection_candidates.append(word)
        elif word in known_reasoning:
            label = "🟢 REASONING (keep)"
        else:
            label = "🟡 AMBIGUOUS (inspect)"
        print(f"  {word:<20} {count:>5}x  {label}")

    # Save keyword list
    keyword_data = {
        "model": args.model_path,
        "datasets": args.datasets,
        "num_samples": args.num_samples,
        "num_runs": args.num_runs,
        "top_words": [{"word": w, "count": c} for w, c in top_words],
        "suggested_suppress": reflection_candidates,
    }

    keyword_file = output_path / "discovered_keywords.json"
    with open(keyword_file, "w") as f:
        json.dump(keyword_data, f, indent=2)
    print(f"\nSaved keyword analysis to {keyword_file}")

    # Also save the raw counts
    counts_file = output_path / "leading_word_counts.csv"
    pd.DataFrame(
        leading_words.most_common(),
        columns=["word", "count"],
    ).to_csv(counts_file, index=False)
    print(f"Saved full word counts to {counts_file}")

    # Save generated texts for inspection
    gen_file = output_path / "generated_cots.parquet"
    pd.DataFrame({
        "problem": problems * (args.num_runs if args.num_runs > 1 else 1),
        "generated": texts,
        "token_count": [len(t.split()) for t in texts],
    }).to_parquet(gen_file)
    print(f"Saved generated CoTs to {gen_file}")

    print(f"\n{'='*60}")
    print(f"  NEXT STEP")
    print(f"{'='*60}")
    print(f"  Review the keywords above, then run evaluation:")
    print(f"")
    print(f"  # Baseline")
    print(f"  python evaluate.py --model-path {args.model_path} \\")
    print(f"      --datasets math-500 gsm8k --mode baseline")
    print(f"")
    print(f"  # NOWAIT with safe keywords")
    print(f"  python evaluate.py --model-path {args.model_path} \\")
    print(f"      --datasets math-500 gsm8k --mode nowait --nowait-keywords safe")
    print(f"")
    print(f"  # NOWAIT with full keywords")
    print(f"  python evaluate.py --model-path {args.model_path} \\")
    print(f"      --datasets math-500 gsm8k --mode nowait --nowait-keywords full")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()