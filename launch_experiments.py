#!/usr/bin/env python3
from __future__ import annotations
import argparse, re, time, os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple
import numpy as np, pandas as pd, torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from math_equivalence import is_equiv

@dataclass
class ModelBundle:
    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    device: torch.device

def load_model_bundle(model_path: str, torch_dtype: torch.dtype = torch.bfloat16) -> ModelBundle:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch_dtype, device_map=device)
    model.eval()
    return ModelBundle(model=model, tokenizer=tokenizer, device=device)

def build_prompt(problem: str, target_think_tokens: int) -> str:
    return f"{problem} Let’s think step by step and output the final answer within boxed{{}}. Think for {target_think_tokens} tokens."

def extract_boxed(s: str):
    m = re.search(r"\\boxed\{([^}]*)\}", s)
    return m.group(1).strip() if m else None

def evaluate_answer(expected_answer, generated_answer):
    exp_val = extract_boxed(expected_answer)
    gen_val = extract_boxed(generated_answer)
    if exp_val is None or gen_val is None:
        return False
    return is_equiv(gen_val, exp_val)

@torch.inference_mode()
def generate_until_eos_batch(model, tokenizer, device, prompts: List[str], hard_cap_new_tokens: Optional[int] = None):
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
    batch_size = inputs.input_ids.size(0)
    outputs = model(**inputs, use_cache=True)
    past = outputs.past_key_values
    generated = inputs.input_ids.clone()
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    new_tokens = torch.zeros(batch_size, dtype=torch.long, device=device)
    while True:
        out = model(input_ids=generated[:, -1:], past_key_values=past, use_cache=True)
        logits = out.logits[:, -1, :]
        past = out.past_key_values
        next_tokens = torch.argmax(logits, dim=-1)
        generated = torch.cat([generated, next_tokens.unsqueeze(-1)], dim=-1)
        new_tokens += (~finished).long()
        eos_mask = next_tokens == tokenizer.eos_token_id
        finished |= eos_mask
        if finished.all():
            break
        if hard_cap_new_tokens is not None and new_tokens.max() >= hard_cap_new_tokens:
            break
    decoded = [tokenizer.decode(seq, skip_special_tokens=True) for seq in generated]
    return decoded, new_tokens.cpu().tolist()

def build_generation_dataset(df, targets, bundle, output_path, batch_size, progress=True):
    model, tokenizer, device = bundle.model, bundle.tokenizer, bundle.device
    outer_iter = tqdm(df.iterrows(), total=len(df), desc="Questions", disable=not progress)
    write_mode = "w" if not os.path.exists(output_path) else "a"
    ext = os.path.splitext(output_path)[1]
    if ext == ".jsonl":
        f = open(output_path, write_mode, encoding="utf-8")
    elif ext == ".csv":
        f = open(output_path, write_mode, newline="", encoding="utf-8")
    else:
        raise ValueError("Use .jsonl or .csv for streaming mode")
    first_write = write_mode == "w"
    try:
        for qid, row in outer_iter:
            problem, solution = str(row["problem"]), str(row["solution"])
            all_prompts = [build_prompt(problem, tgt) for tgt in targets]
            all_targets = list(targets)
            for b in range(0, len(all_prompts), batch_size):
                batch_prompts = all_prompts[b : b + batch_size]
                batch_targets = all_targets[b : b + batch_size]
                t0 = time.perf_counter()
                gen_texts, gen_counts = generate_until_eos_batch(model, tokenizer, device, batch_prompts)
                t1 = time.perf_counter()
                for prompt, tgt, gen_text, total_tokens in zip(batch_prompts, batch_targets, gen_texts, gen_counts):
                    record = {
                        "question_id": qid,
                        "prompt": prompt,
                        "solution_col": solution,
                        "generated_text": gen_text,
                        "target_think_tokens": int(tgt),
                        "generated_total_tokens": int(total_tokens),
                        "latency_sec": float(t1 - t0),
                        "is_correct": bool(evaluate_answer(solution, gen_text)),
                    }
                    if ext == ".jsonl":
                        f.write(pd.Series(record).to_json(force_ascii=False) + "\n")
                    else:
                        pd.DataFrame([record]).to_csv(f, header=first_write, index=False)
                        first_write = False
                    f.flush()
            break
    finally:
        f.close()

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--no-progress", action="store_true")
    return p.parse_args()

def main():
    args = parse_args()
    df = pd.read_parquet(args.data)
    targets = np.linspace(start=100, stop=2500, num=10, endpoint=True, dtype=int)
    bundle = load_model_bundle(args.model_path)
    build_generation_dataset(df, targets, bundle, args.output, progress=not args.no_progress, batch_size=1)

if __name__ == "__main__":
    main()
