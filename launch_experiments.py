#!/usr/bin/env python3
from __future__ import annotations
import argparse, re, time, os
from dataclasses import dataclass
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
    return f"{problem} Let’s think step by step inside and output the final answer within boxed{{}}. Think for {target_think_tokens} tokens. <think>"

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

def decode_returns(tokenizer, generated):
    decoded = [tokenizer.decode(seq, skip_special_tokens=True) for seq in generated]
    think_texts, think_token_counts = [], []
    for text in decoded:
        think_text = extract_think_text(text)
        if think_text:
            think_ids = tokenizer(think_text, return_tensors="pt").input_ids
            think_token_counts.append(int(think_ids.shape[1]))
        else:
            think_token_counts.append(0)
        think_texts.append(think_text)
    return think_texts,think_token_counts

def save_hidden_states_step(hidden_records, hidden_last, generated, token_counters, next_save_step, save_active, end_think_ids, activation_interval):
    batch_size = hidden_last.size(0)
    len_end = len(end_think_ids)

    for i in range(batch_size):
        if not save_active[i]:
            continue  # already stopped saving after </think>

        # Check if </think> appeared in this sample
        gen_ids = generated[i].tolist()
        if len(gen_ids) >= len_end and gen_ids[-len_end:] == end_think_ids:
            save_active[i] = False  # stop saving from now on
            continue

        # Save hidden states at defined intervals
        if token_counters[i] >= next_save_step[i]:
            hidden_records.append({
                "sample_idx": int(i),
                "token_step": int(token_counters[i].item()),
                "hidden_state": hidden_last[i].detach().to(torch.float16).cpu().numpy().tolist(),
            })
            next_save_step[i] += activation_interval

@torch.inference_mode()
def generate_until_eos_batch(model, tokenizer,device,prompts,activation_interval):
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
    batch_size = inputs.input_ids.size(0)

    outputs = model(**inputs, use_cache=True)
    past = outputs.past_key_values
    generated = inputs.input_ids.clone()
    hidden_records = []
    
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    token_counters = torch.zeros(batch_size, dtype=torch.long, device=device)
    next_save_step = torch.full((batch_size,), activation_interval, dtype=torch.long, device=device)
    save_active = torch.ones(batch_size, dtype=torch.bool, device=device)  # saving initially active
    end_think_ids = tokenizer("</think>", add_special_tokens=False).input_ids

    max_new_tokens = 4096
    for _ in range(max_new_tokens):
        out = model(
            input_ids=generated[:, -1:],
            past_key_values=past,
            use_cache=True,
            output_hidden_states=True,
        )

        logits = out.logits[:, -1, :]
        past = out.past_key_values
        hidden_last = out.hidden_states[-1][:, -1, :]  # [batch, hidden_dim]

        next_tokens = torch.argmax(logits, dim=-1)
        generated = torch.cat([generated, next_tokens.unsqueeze(-1)], dim=-1)

        eos_mask = next_tokens == tokenizer.eos_token_id
        finished |= eos_mask

        token_counters += (~finished).long()

        save_hidden_states_step(
            hidden_records=hidden_records,
            hidden_last=hidden_last,
            generated=generated,
            token_counters=token_counters,
            next_save_step=next_save_step,
            save_active=save_active,
            end_think_ids=end_think_ids,
            activation_interval=activation_interval,
        )
        if finished.all():
            break
        
    if not finished.all():
        print("Warning: stopped by max_new_tokens (no EOS emitted).")
    think_texts, think_token_counts = decode_returns(tokenizer, generated)
    return [tokenizer.decode(seq, skip_special_tokens=True) for seq in generated],think_texts, think_token_counts, hidden_records


def build_generation_dataset(df, targets, bundle, generated_dir, hidden_dir, batch_size, progress=True):
    os.makedirs(generated_dir, exist_ok=True)
    os.makedirs(hidden_dir, exist_ok=True)

    model, tokenizer, device = bundle.model, bundle.tokenizer, bundle.device
    batch_id = 1

    # Each question expands into 10 prompts (1 per target)
    expanded = []
    for qid, row in df.iterrows():
        problem, solution = str(row["problem"]), str(row["solution"])
        for tgt in targets:
            expanded.append({
                "question_id": qid,
                "problem": problem,
                "solution": solution,
                "target_think_tokens": int(tgt),
            })
    expanded_df = pd.DataFrame(expanded)

    for b in tqdm(range(0, len(expanded_df), batch_size), desc="Global batches", disable=not progress):
        batch = expanded_df.iloc[b : b + batch_size]
        batch_prompts = [build_prompt(p, t) for p, t in zip(batch["problem"], batch["target_think_tokens"])]
        batch_targets = batch["target_think_tokens"].tolist()
        batch_qids = batch["question_id"].tolist()
        batch_solutions = batch["solution"].tolist()

        t0 = time.perf_counter()
        full_texts, think_texts, think_counts, hidden_records = generate_until_eos_batch(
            model, tokenizer, device, batch_prompts, activation_interval=50
        )
        t1 = time.perf_counter()

        gen_records = []
        for qid, tgt, prompt, full_text, think_text, think_tokens, sol in zip(
            batch_qids, batch_targets, batch_prompts, full_texts, think_texts, think_counts, batch_solutions
        ):
            is_ok = evaluate_answer(sol, full_text)
            gen_records.append({
                "question_id": qid,
                "prompt": prompt,
                "solution_col": sol,
                "generated_think_text": think_text,
                "generated_text": full_text,
                "target_think_tokens": int(tgt),
                "generated_think_tokens": int(think_tokens),
                "latency_sec": float(t1 - t0),
                "is_correct": bool(is_ok),
            })

        gen_df = pd.DataFrame(gen_records)
        gen_path = os.path.join(generated_dir, f"generated_batch{batch_id}.csv")
        gen_df.to_csv(gen_path, index=False)

        # === Save hidden info + states ===
        if hidden_records:
            meta, states = [], []
            for rec in hidden_records:
                idx = rec["sample_idx"]
                meta.append({
                    "question_id": batch_qids[idx],
                    "sample_idx": idx,
                    "token_step": rec["token_step"],
                    "target_think_tokens": batch_targets[idx],
                    "is_correct": evaluate_answer(batch_solutions[idx], full_texts[idx]),
                })
                states.append(np.array(rec["hidden_state"], dtype=np.float16))

            meta_df = pd.DataFrame(meta)
            meta_path = os.path.join(hidden_dir, f"hidden_info_batch{batch_id}.csv")
            arr_path = os.path.join(hidden_dir, f"hidden_numpyarr_batch{batch_id}.npy")

            meta_df.to_csv(meta_path, index=False)
            np.save(arr_path, np.stack(states, axis=0))

        batch_id += 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--generated-dir", required=True)
    p.add_argument("--hidden-dir",required=True)
    return p.parse_args()

def main():
    args = parse_args()
    df = pd.read_parquet(args.data)
    targets = np.linspace(start=100, stop=2500, num=10, endpoint=True, dtype=int)
    bundle = load_model_bundle(args.model_path)
    build_generation_dataset(df, targets, bundle, args.generated_dir, args.hidden_dir, batch_size=4)
    
if __name__ == "__main__":
    main()
