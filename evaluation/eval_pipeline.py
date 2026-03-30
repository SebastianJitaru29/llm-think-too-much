import argparse
import re
import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from data.processing.math_equivalence import is_equiv
from evaluation.evaluation import evaluate_answer

EVAL_DATASET_PATH = Path(__file__).parent.parent / "data" / "raw" / "eval_data.parquet"

# 17 keywords from the paper (Table in Section 3.1)
NOWAIT_KEYWORDS = [
    "wait", "alternatively", "hmm", "but", "however",
    "alternative", "another", "check", "double-check",
    "oh", "maybe", "verify", "other", "again", "now", "ah", "any",
]


def build_suppress_ids(tokenizer, keywords: list[str]) -> list[int]:
    """Scan vocabulary, collect token IDs whose decoded text matches a keyword."""
    suppress = set()
    for token_id in range(tokenizer.vocab_size):
        try:
            decoded = tokenizer.decode([token_id])
        except Exception:
            continue
        cleaned = decoded.strip().lower()
        if not cleaned:
            continue
        for kw in keywords:
            if kw in cleaned:
                suppress.add(token_id)
                break
    return sorted(suppress)


def extract_think_text(full_text: str) -> str:
    match = re.search(r"<think>(.*?)</think>", full_text, flags=re.DOTALL)
    return match.group(1).strip() if match else ""


def get_token_budget_prompt(question: str, tokenizer) -> str:
    """Build a prompt asking the model to estimate its own token budget (TALE)."""
    prompt = (
        f"Q: {question}"
        "Task: Analyze the given question and estimate the minimum number of tokens "
        "required to generate a complete and accurate response. "
        "Please Give the response by strictly following this format: [[budget]]. "
        "for example, Budget: [[12]].\n\n"
    )
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )


def build_prompt_content(problem: str, dataset_name: str) -> str:
    """Build the user-facing problem content (shared across modes)."""
    if dataset_name == "aime-250":
        return problem
    return f"{problem}\nLet's think step by step and output the final answer within boxed{{}}."


def format_prompt(content: str, tokenizer):
    """Apply chat template to content."""
    messages = [{"role": "user", "content": content}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )


def run_tale_budget_estimation(problems, llm, tokenizer, sampling_params, save_dir=None):
    """Stage 1 of TALE: ask the model to estimate token budgets."""
    budget_prompts = [get_token_budget_prompt(p, tokenizer) for p in problems]

    outputs = llm.generate(budget_prompts, sampling_params)

    targets = []
    output_texts = []
    for output in outputs:
        if len(output.outputs) > 0:
            text = output.outputs[0].text
            output_texts.append(text)
            match = re.search(r"\[\[(\d+)\]\]", text)
            targets.append(int(match.group(1)) if match else 100)
        else:
            output_texts.append("")
            targets.append(100)

    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        np.save(save_path / "tale_budget_estimates.npy", {
            "targets": np.array(targets),
            "outputs": np.array(output_texts),
        })
        print(f"Saved TALE budget estimates to {save_path / 'tale_budget_estimates.npy'}")

    return targets


def run_dataset(args, subset, dataset_name, llm, tokenizer, logit_bias=None, tale_targets=None, regressor_targets=None):
    """Run a single dataset through the chosen mode. Returns list of result dicts."""
    mode = args.mode

    # --- Build prompts ---
    prompts = []
    target_tokens_list = []
    for i, (_, row) in enumerate(subset.iterrows()):
        content = build_prompt_content(row["problem"], dataset_name)

        if mode == "regressor":
            target = int(regressor_targets[i])
            content = f"{content} Think for {target} tokens."
            target_tokens_list.append(target)
            prompts.append(format_prompt(content, tokenizer))
        elif mode == "sentence":
            target = int(regressor_targets[i])
            content = f"{content} Use less than {target // 60} sentences."
            target_tokens_list.append(target)
            prompts.append(format_prompt(content, tokenizer))
        elif mode == "tale":
            target = int(tale_targets[i])
            content = f"{content} Use less than {target} tokens."
            target_tokens_list.append(target)
            prompts.append(format_prompt(content, tokenizer))
        else:
            prompts.append(format_prompt(content, tokenizer))

    # --- Sampling params ---
    sampling_params = SamplingParams(
        max_tokens=32000,
        temperature=0,
        skip_special_tokens=False,
        logit_bias=logit_bias,
    )

    # --- Generate ---
    t0 = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    t1 = time.perf_counter()

    # --- Collect results ---
    results = []
    correct = 0
    total_tokens = 0
    for i, output in enumerate(outputs):
        text = output.outputs[0].text
        token_count = len(output.outputs[0].token_ids)
        total_tokens += token_count

        is_correct, exp_val, gen_val = evaluate_answer(
            str(subset.iloc[i]["solution"]), text
        )
        if is_correct:
            correct += 1

        results.append({
            "unique_id": subset.iloc[i]["unique_id"],
            "prompt": prompts[i],
            "solution": subset.iloc[i]["solution"],
            "generated": text,
            "expected_value": exp_val,
            "generated_value": gen_val,
            "token_count": token_count,
            "is_correct": is_correct,
            "latency_sec": (t1 - t0) / len(prompts),
        })

    acc = correct / len(subset)
    avg_tok = total_tokens / len(subset)
    print(f"{dataset_name:<15} acc={acc*100:.1f}%  avg_tokens={avg_tok:.0f}  ({correct}/{len(subset)})")

    return results


def main():
    parser = argparse.ArgumentParser(description="Unified evaluation pipeline")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--datasets", type=str, nargs="+", default=["math-500", "gsm8k", "olympiad"])
    parser.add_argument("--output-dir", type=str, default="./experiments")
    parser.add_argument("--mode", type=str, required=True,
                        choices=["baseline", "nowait", "regressor", "tale", "sentence"],
                        help="Evaluation mode")
    parser.add_argument("--target-tokens", type=str, default=None,
                        help="Path to .npy with target tokens (required for regressor/sentence, optional for tale)")
    parser.add_argument("--max-model-len", type=int, default=32000)
    args = parser.parse_args()

    if args.mode in ("regressor", "sentence") and args.target_tokens is None:
        parser.error("--target-tokens is required for regressor/sentence mode")

    # --- Load data (always eval_data.parquet) ---
    df = pd.read_parquet(EVAL_DATASET_PATH)
    df = df[df["dataset"].isin(args.datasets)]
    df["unique_id"] = df["id"].astype(str) + "-" + df["dataset"]
    print(f"Loaded {len(df)} problems: {df['dataset'].value_counts().to_dict()}")

    # --- Load model ---
    llm = LLM(
        model=args.model_path, dtype="bfloat16", trust_remote_code=True,
        max_model_len=args.max_model_len,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # --- Mode-specific setup ---
    logit_bias = None
    if args.mode == "nowait":
        suppress_ids = build_suppress_ids(tokenizer, NOWAIT_KEYWORDS)
        print(f"Suppressing {len(suppress_ids)} token IDs from {len(NOWAIT_KEYWORDS)} keywords")
        logit_bias = {tid: -100.0 for tid in suppress_ids}

    regressor_targets_map = None
    if args.mode in ("regressor", "sentence"):
        data = np.load(args.target_tokens, allow_pickle=True).item()
        regressor_targets_map = dict(zip(data["ids"], data["target"]))
        print(f"Loaded regressor targets for {len(regressor_targets_map)} problems")

    # --- Run per dataset ---
    run_name = f"{Path(args.model_path).stem}_{args.mode}"
    output_path = Path(args.output_dir) / run_name
    output_path.mkdir(parents=True, exist_ok=True)

    for dataset_name in args.datasets:
        subset = df[df["dataset"] == dataset_name].reset_index(drop=True)
        if len(subset) == 0:
            continue

        # Build per-row target tokens for regressor/tale
        regressor_targets = None
        tale_targets = None

        if args.mode in ("regressor", "sentence"):
            regressor_targets = []
            for _, row in subset.iterrows():
                key = row["id"]
                if key not in regressor_targets_map:
                    raise ValueError(f"No regressor target for problem id={key}")
                regressor_targets.append(regressor_targets_map[key])

        if args.mode == "tale":
            if args.target_tokens:
                tale_data = np.load(args.target_tokens, allow_pickle=True).item()
                tale_targets = list(tale_data["targets"])[:len(subset)]
                print(f"Loaded {len(tale_targets)} pre-computed TALE targets for {dataset_name}")
            else:
                print(f"TALE stage 1: estimating token budgets for {dataset_name}...")
                problems = [row["problem"] for _, row in subset.iterrows()]
                save_dir = output_path
                tale_sampling = SamplingParams(
                    max_tokens=20_000, temperature=0, skip_special_tokens=False,
                )
                tale_targets = run_tale_budget_estimation(
                    problems, llm, tokenizer, tale_sampling, save_dir,
                )
                print(f"Estimated budgets — min={min(tale_targets)}, max={max(tale_targets)}, mean={np.mean(tale_targets):.0f}")

        results = run_dataset(
            args, subset, dataset_name, llm, tokenizer,
            logit_bias=logit_bias,
            tale_targets=tale_targets,
            regressor_targets=regressor_targets,
        )
        pd.DataFrame(results).to_parquet(output_path / f"{dataset_name}_results.parquet")

    print(f"\nResults saved to {output_path}")

    del llm
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
