#!/usr/bin/env python3
from __future__ import annotations
import argparse, re, time, os
from dataclasses import dataclass
import numpy as np, pandas as pd, torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from math_equivalence import is_equiv
from transformers import StoppingCriteria, StoppingCriteriaList
import re



@dataclass
class ModelBundle:
    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    device: torch.device


def load_model_bundle(model_path: str, torch_dtype: torch.dtype = torch.bfloat16) -> ModelBundle:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    device = torch.device("cuda")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map=device,
    )
    model.eval()
    return ModelBundle(model, tokenizer, device)


def build_prompt(problem: str, target_think_tokens: int) -> str:
    return f"{problem} Let’s think step by step inside and output the final answer within boxed{{}}. Think for {target_think_tokens} tokens."


def extract_boxed(s: str):
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




# ------------------------------------------------------------
# Regex stopping criterion (batch size = 1)
# ------------------------------------------------------------
class RegexStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, pattern: str):
        self.tokenizer = tokenizer
        self.regex = re.compile(pattern)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs):
        # Assumes batch size = 1
        seq = input_ids[0]
        text = self.tokenizer.decode(seq, skip_special_tokens=False)

        # Stop if regex is detected
        return bool(self.regex.search(text))


# ------------------------------------------------------------
# Helper: truncate text at first \boxed{...}
# ------------------------------------------------------------
def truncate_after_boxed(text: str):
    m = re.search(r"\\boxed\{[^}]*\}", text)
    if m:
        return text[:m.end()]
    return text


# ------------------------------------------------------------
# Batched generation via HF (but stopping only checks first sample)
# ------------------------------------------------------------
@torch.inference_mode()
def generate_batch_hf(model, tokenizer, device, prompts, max_new_tokens=6000):

    # enforce batch size = 1 for reliable regex stopping
    if len(prompts) != 1:
        raise ValueError("RegexStoppingCriteria only works when len(prompts) == 1")

    # regex for \boxed{something}
    stop_regex = r"\\boxed\{[^}]*\}"

    stopping = StoppingCriteriaList([
        RegexStoppingCriteria(tokenizer, stop_regex)
    ])

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        stopping_criteria=stopping
    )

    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=False)

    # Truncate everything after first \boxed{...}
    decoded = [truncate_after_boxed(t) for t in decoded]

    return decoded



# ----------------------------------------------------------------------
# Dataset building
# ----------------------------------------------------------------------

def build_generation_dataset(df, targets, bundle, generated_dir, batch_size, progress=True):
    os.makedirs(generated_dir, exist_ok=True)

    model, tokenizer, device = bundle.model, bundle.tokenizer, bundle.device
    batch_id = 1
    part_id = 1  # parquet file counter

    # accumulator for 100 batches
    accumulated_rows = []

    # expand rows
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

    for b in tqdm(range(0, len(expanded_df), batch_size), desc="Global batches", disable=not progress):
        batch = expanded_df.iloc[b: b + batch_size]

        batch_prompts = [
            build_prompt(p, t)
            for p, t in zip(batch["problem"], batch["target_think_tokens"])
        ]

        t0 = time.perf_counter()
        full_texts = generate_batch_hf(model, tokenizer, device, batch_prompts)
        t1 = time.perf_counter()

        records = []
        for (qid, sol, tgt, prompt, full_text) in zip(
            batch["question_id"],
            batch["solution"],
            batch["target_think_tokens"],
            batch_prompts,
            full_texts,
        ):
            think_text = extract_think_text(full_text)
            think_token_count = (
                tokenizer(think_text, return_tensors="pt").input_ids.shape[1]
                if think_text else 0
            )

            is_ok = evaluate_answer(sol, full_text)

            records.append({
                "question_id": qid,
                "prompt": prompt,
                "solution_col": sol,
                "generated_think_text": think_text,
                "generated_text": full_text,
                "target_think_tokens": int(tgt),
                "generated_think_tokens": int(think_token_count),
                "latency_sec": float(t1 - t0),
                "is_correct": bool(is_ok),
            })

        accumulated_rows.extend(records)

        # ------------------------------------------------------------
        # Save every 100 batches
        # ------------------------------------------------------------
        if batch_id % 100 == 0:
            out_df = pd.DataFrame(accumulated_rows)
            out_path = os.path.join(generated_dir, f"generated_part{part_id}.parquet")
            out_df.to_parquet(out_path, index=False)
            accumulated_rows = []  # reset accumulator
            part_id += 1

        batch_id += 1

    # ------------------------------------------------------------
    # Save remainder (if fewer than 100 batches left)
    # ------------------------------------------------------------
    if accumulated_rows:
        out_df = pd.DataFrame(accumulated_rows)
        out_path = os.path.join(generated_dir, f"generated_part{part_id}.parquet")
        out_df.to_parquet(out_path, index=False)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--generated-dir", required=True)
    p.add_argument("--batch-size", type=int, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    df = pd.read_parquet(args.data)
    targets = np.linspace(start=100, stop=5500, num=20, endpoint=True, dtype=int)
    bundle = load_model_bundle(args.model_path)
    build_generation_dataset(df, targets, bundle, args.generated_dir, args.batch_size)


if __name__ == "__main__":
    main()
