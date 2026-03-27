"""NoWait: suppress reflection tokens during inference.

Replicates Wang et al. "Wait, We Don't Need to Wait!" (arXiv:2506.08343).
The method sets logits of self-reflection keywords to -inf during decoding,
reducing CoT length by 27-51% with minimal accuracy loss on RL-trained models.
"""

import argparse
import re
import gc
from pathlib import Path

import pandas as pd
import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from data.processing.math_equivalence import is_equiv

EVAL_DATASET_PATH = Path(__file__).parent.parent.parent.parent / "data" / "raw" / "eval_data.parquet"

# 17 keywords from the paper (Table in Section 3.1)
NOWAIT_KEYWORDS = [
    "wait", "alternatively", "hmm", "but", "however",
    "alternative", "another", "check", "double-check",
    "oh", "maybe", "verify", "other", "again", "now", "ah", "any",
]


def build_suppress_ids(tokenizer, keywords: list[str]) -> list[int]:
    """Scan vocabulary, collect token IDs whose decoded text matches a keyword.

    Uses substring matching as described in the paper: a token is suppressed
    if any keyword is a substring of the token's decoded text (case-insensitive).
    """
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


def make_logits_processor(suppress_ids: list[int]):
    def processor(token_ids, logits):
        logits[suppress_ids] = float("-inf")
        return logits
    return processor


def extract_boxed(s: str) -> str | None:
    if not s:
        return None
    matches = re.findall(r"\\{1,2}boxed\{([^}]*)\}", s)
    if matches:
        return matches[-1].strip()
    matches = re.findall(r"(?m)^[ \t]*####[ \t]*([^\n\r#]+?)[ \t]*$", s)
    if matches:
        return matches[-1].strip()
    matches = re.findall(r"\$([^$]*)\$", s)
    if matches:
        return matches[-1].strip()
    matches = re.findall(r"(?m)^[ \t]*([+-]?\d+(?:\.\d+)?)[ \t]*$", s)
    if matches:
        return matches[-1].strip()
    return s


def evaluate_answer(expected: str, generated: str):
    exp = extract_boxed(expected)
    gen = extract_boxed(generated)
    if exp is None or gen is None:
        return False, exp, gen
    return is_equiv(gen, exp), exp, gen


def main():
    parser = argparse.ArgumentParser(description="NoWait evaluation")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--datasets", type=str, nargs="+", default=["math-500", "gsm8k", "olympiad"])
    parser.add_argument("--output-dir", type=str, default="./eval_results")
    parser.add_argument("--mode", type=str, default="nowait", choices=["baseline", "nowait"])
    args = parser.parse_args()

    df = pd.read_parquet(EVAL_DATASET_PATH)
    df = df[df["dataset"].isin(args.datasets)]
    df["unique_id"] = df["id"].astype(str) + "-" + df["dataset"]
    print(f"Loaded {len(df)} problems: {df['dataset'].value_counts().to_dict()}")

    llm = LLM(model=args.model_path, dtype="bfloat16", trust_remote_code=True, max_model_len=32000)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    logits_processors = []
    if args.mode == "nowait":
        suppress_ids = build_suppress_ids(tokenizer, NOWAIT_KEYWORDS)
        print(f"Suppressing {len(suppress_ids)} token IDs from {len(NOWAIT_KEYWORDS)} keywords")
        logits_processors.append(make_logits_processor(suppress_ids))

    sampling_params = SamplingParams(
        max_tokens=32000,
        temperature=0,
        skip_special_tokens=False,
        logits_processors=logits_processors or None,
    )

    run_name = f"{Path(args.model_path).stem}_{args.mode}"
    output_path = Path(args.output_dir) / run_name
    output_path.mkdir(parents=True, exist_ok=True)

    for dataset_name in args.datasets:
        subset = df[df["dataset"] == dataset_name]
        if len(subset) == 0:
            continue

        prompts = []
        for _, row in subset.iterrows():
            if dataset_name == "aime-250":
                prompts.append(row["problem"])
            else:
                prompts.append(f"{row['problem']}\nLet's think step by step and output the final answer within boxed{{}}.")
        outputs = llm.generate(prompts, sampling_params)

        results = []
        correct = 0
        total_tokens = 0
        for i, output in enumerate(outputs):
            text = output.outputs[0].text
            token_count = len(output.outputs[0].token_ids)
            total_tokens += token_count
            is_correct, exp_val, gen_val = evaluate_answer(str(subset.iloc[i]["solution"]), text)
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
            })

        acc = correct / len(subset)
        avg_tok = total_tokens / len(subset)
        print(f"{dataset_name:<15} acc={acc*100:.1f}%  avg_tokens={avg_tok:.0f}  ({correct}/{len(subset)})")

        pd.DataFrame(results).to_parquet(output_path / f"{dataset_name}_results.parquet")

    del llm
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
