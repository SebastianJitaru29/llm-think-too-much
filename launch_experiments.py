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
    decoded = [tokenizer.decode(seq, skip_special_tokens=False) for seq in generated]
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
    return [tokenizer.decode(seq, skip_special_tokens=False) for seq in generated],think_texts, think_token_counts, hidden_records


def build_generation_dataset(df, targets, bundle, output_path, hidden_path, batch_size, progress=True):
    model, tokenizer, device = bundle.model, bundle.tokenizer, bundle.device

    write_mode = "w" if not os.path.exists(output_path) else "a"
    f_main = open(output_path, write_mode, newline="", encoding="utf-8")
    f_hidden = open(hidden_path, write_mode, newline="", encoding="utf-8")

    first_write_main = write_mode == "w"
    first_write_hidden = write_mode == "w"

    try:
        for qid, row in tqdm(df.iterrows(), total=len(df), desc="Questions", disable=not progress):
            problem, solution = str(row["problem"]), str(row["solution"])
            all_prompts = [build_prompt(problem, tgt) for tgt in targets]
            all_targets = list(targets)

            for b in range(0, len(all_prompts), batch_size):
                batch_prompts = all_prompts[b:b + batch_size]
                batch_targets = all_targets[b:b + batch_size]

                t0 = time.perf_counter()
                full_texts, think_texts, think_counts, hidden_records = generate_until_eos_batch(model, tokenizer, device, batch_prompts, 50)
                t1 = time.perf_counter()

                records = []
                for prompt, tgt, full_text, think_text, think_tokens in zip(batch_prompts, batch_targets, full_texts,think_texts, think_counts):
                    is_ok = evaluate_answer(solution, full_text)
                    records.append({
                        "question_id": qid,
                        "prompt": prompt,
                        "solution_col": solution,
                        "generated_think_text": think_text,
                        "generated_text": full_text,
                        "target_think_tokens": int(tgt),
                        "generated_think_tokens": int(think_tokens),
                        "latency_sec": float(t1 - t0),
                        "is_correct": bool(is_ok),
                    })

                pd.DataFrame(records).to_csv(f_main, header=first_write_main, index=False)
                first_write_main = False
                f_main.flush()

                if hidden_records:
                    hidden_records_aug = []
                    for rec in hidden_records:
                        # Map hidden sample to current question and target run
                        sample_idx = rec.get("sample_idx", 0)
                        tgt = batch_targets[sample_idx] if sample_idx < len(batch_targets) else None
                        is_ok = evaluate_answer(solution, full_texts[sample_idx]) if sample_idx < len(full_text) else False   
                        hidden_records_aug.append({
                            "question_id": qid,
                            "target_think_tokens": int(tgt),
                            "token_step": int(rec["token_step"]),
                            "is_correct": bool(is_ok),
                            "hidden_state": rec["hidden_state"],
                        })

                    pd.DataFrame(hidden_records_aug).to_csv(f_hidden, header=first_write_hidden, index=False)
                    first_write_hidden = False
                    f_hidden.flush()

    finally:
        f_main.close()
        f_hidden.close()

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--outputhidden",required=True)
    return p.parse_args()

def main():
    args = parse_args()
    df = pd.read_parquet(args.data)
    targets = np.linspace(start=100, stop=2500, num=10, endpoint=True, dtype=int)
    bundle = load_model_bundle(args.model_path)
    build_generation_dataset(df, targets, bundle, args.output, args.outputhidden, batch_size=64)
    
if __name__ == "__main__":
    main()
