
import argparse
from pathlib import Path
import os
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
import shutil

from regressor.architecture import Regressor
from launch_experiments import (
    build_prompt,
    extract_boxed,
    evaluate_answer,
    extract_think_text,
    load_model_bundle,
    ModelBundle
)

def load_jax_and_set_to_cpu() -> Regressor:
    jax.default_device(jax.devices('cpu')[0])
    network = Regressor.load_network()
    return network

@jax.jit
def predict_batch(network: Regressor, hidden: jax.Array) -> tuple[jax.Array, jax.Array]:
    logits = Regressor.forward(hidden, network)
    p = jax.nn.sigmoid(logits)
    bins_correct = p > 0.5
    
    any_correct = jnp.any(bins_correct, axis=1)
    highest_incorrect = jnp.argmax(p, axis=1)
    min_correct = jnp.argmax(bins_correct, axis=1)

    bucket_i = jax.lax.select(any_correct, min_correct, highest_incorrect)

    return bucket_i, p


def numpy_wrapper_predict_batch(network: Regressor, hidden: torch.Tensor) -> tuple[list[int], np.ndarray]:
    
    hidden = jax.device_put(hidden.to(torch.float32).cpu().numpy())
    bucket_i, p = predict_batch(network, hidden)
    target_tokens = []

    for bi in bucket_i:
        target_tokens.append(network.bins[bi])
    
    assert len(target_tokens) == hidden.shape[0]

    p = np.array(p)

    return target_tokens, p

def load_questions(
    train: bool = True
):
    df = pd.read_parquet(Path(__file__).parent / "data" / "dataset.parquet")

    if not train:
        return df.sample(4)
    return df

@torch.inference_mode()
def get_initial_hidden_states(model, tokenizer, device, problems: list[str]) -> torch.Tensor:
    """Get initial hidden states for a batch of problems (before adding 'Think for X tokens')."""
    # Create simple prompts with just the problem and instruction
    simple_prompts = [
        f"{problem}"
        for problem in problems
    ]
    
    inputs = tokenizer(simple_prompts, return_tensors="pt", padding=True, truncation=True).to(device)
    outputs = model(**inputs, output_hidden_states=True)
    # Get last hidden state of the last layer for the last token
    hidden_states = outputs.hidden_states[-1][:, -1, :]  # [batch, hidden_dim]
    return hidden_states.cpu()

@torch.inference_mode()
def generate_with_prompt(model, tokenizer, device, prompts: list[str], max_new_tokens: int = 4096):
    """Generate text for a batch of prompts and return full text and token counts."""
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
    batch_size = inputs.input_ids.size(0)
    
    outputs = model(**inputs, use_cache=True)
    past = outputs.past_key_values
    generated = inputs.input_ids.clone()
    
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    token_counters = torch.zeros(batch_size, dtype=torch.long, device=device)
    
    for _ in range(max_new_tokens):
        out = model(
            input_ids=generated[:, -1:],
            past_key_values=past,
            use_cache=True,
        )
        
        logits = out.logits[:, -1, :]
        past = out.past_key_values
        
        next_tokens = torch.argmax(logits, dim=-1)
        generated = torch.cat([generated, next_tokens.unsqueeze(-1)], dim=-1)
        
        eos_mask = next_tokens == tokenizer.eos_token_id
        finished |= eos_mask
        token_counters += (~finished).long()
        
        if finished.all():
            break
    
    # Decode generated sequences
    full_texts = [tokenizer.decode(seq, skip_special_tokens=False) for seq in generated]
    
    # Extract think text and count tokens
    think_texts = []
    think_token_counts = []
    for text in full_texts:
        think_text = extract_think_text(text)
        if think_text:
            think_ids = tokenizer(think_text, return_tensors="pt").input_ids
            think_token_counts.append(int(think_ids.shape[1]))
        else:
            think_token_counts.append(0)
        think_texts.append(think_text)
    
    return full_texts, think_token_counts


import re

@torch.inference_mode()
def generate_partially_with_prompt(
    model,
    tokenizer,
    device,
    prompts: list[str],
    n_tokens: int = 128
):
    """Generate up to n_tokens and return text, token counts, and final hidden states."""
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
    batch_size = inputs.input_ids.size(0)
    generated = inputs.input_ids.clone()
    token_counters = torch.zeros(batch_size, dtype=torch.long, device=device)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    past = None

    for _ in range(n_tokens):
        out = model(
            input_ids=generated[:, -1:] if past is not None else generated,
            past_key_values=past,
            use_cache=True,
        )
        logits = out.logits[:, -1, :]
        past = out.past_key_values

        next_tokens = torch.argmax(logits, dim=-1)
        generated = torch.cat([generated, next_tokens.unsqueeze(-1)], dim=-1)
        eos_mask = next_tokens == tokenizer.eos_token_id
        finished |= eos_mask
        token_counters += (~finished).long()

        if finished.all():
            break

    full_texts = [tokenizer.decode(seq, skip_special_tokens=False) for seq in generated]
    hidden_states = out.hidden_states[-1][:, -1, :].detach().cpu()  # latest hidden state
    return full_texts, token_counters.tolist(), hidden_states, finished.cpu().tolist()


def change_target_tokens(prompt: str, new_target: int) -> str:
    """Replace the target token count in the prompt with the new target."""
    return re.sub(r"Think for \d+ tokens", f"Think for {new_target} tokens", prompt)


def test_inference_dynamic(
    output_path: Path,
    model_path: Path = Path(__file__).parent / "models" / "L1-Qwen-1.5B-Exact",
    batch_size: int = 2,
    n_tokens: int = 100,
    max_steps: int = 500
):
    print("Loading test questions...")
    test_df = load_questions(train=False)

    print("Loading regressor network...")
    network = Regressor.load_network()
    bins = network.bins

    print(f"Loading L1 model from {model_path}...")
    bundle = load_model_bundle(model_path)
    model, tokenizer, device = bundle.model, bundle.tokenizer, bundle.device

    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    num_batches = int(np.ceil(len(test_df) / batch_size))
    for batch_idx in tqdm(range(num_batches), desc="Processing dynamic batches"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(test_df))
        batch_df = test_df.iloc[start_idx:end_idx]

        problems = batch_df["problem"].tolist()
        solutions = batch_df["solution"].tolist()
        question_ids = batch_df.index.tolist()

        initial_hidden = get_initial_hidden_states(model, tokenizer, device, problems)
        predicted_targets, targets_p = numpy_wrapper_predict_batch(network, initial_hidden)
        full_prompts = [build_prompt(p, t) for p, t in zip(problems, predicted_targets, strict=True)]
        partial_texts = ["" for _ in range(len(problems))]

        all_results = []

        for step_i in range(max_steps):
            print(step_i)
            generated_texts, token_counts, latest_hidden, finished_mask  = generate_partially_with_prompt(
                model, tokenizer, device, full_prompts, n_tokens=n_tokens
            )

            # Append new generation to existing text
            partial_texts = [
                old + new for old, new in zip(partial_texts, generated_texts, strict=True)
            ]

            # Evaluate updated hidden and predictions
            predicted_targets, targets_p = numpy_wrapper_predict_batch(network, latest_hidden)

            # Evaluate correctness only if finished
            for qid, problem, solution, target, actual, gen_text, target_p, finished in zip(
                question_ids, problems, solutions, predicted_targets, token_counts, partial_texts, targets_p, finished_mask, strict=True
            ):
                is_correct = evaluate_answer(solution, gen_text) if finished else np.nan
                row = {
                    "question_id": qid,
                    "step_i": step_i,
                    "problem": problem,
                    "expected_solution": solution,
                    "target_tokens": target,
                    "actual_tokens": actual,
                    "is_correct": is_correct,
                    "generated_text": gen_text,
                }
                for b, p in zip(bins, target_p, strict=True):
                    row[f"bin_{b}"] = p
                all_results.append(row)

            # Break early if all sequences finished
            if all(finished_mask):
                break

            # Update prompts for next step (if any unfinished)
            full_prompts = [
                change_target_tokens(prompt, t)
                for prompt, t in zip(full_prompts, predicted_targets, strict=True)
            ]

        results_df = pd.DataFrame(all_results)
        results_df.to_parquet(output_path / f"batch_{batch_idx}.parquet", index=False)

    return results_df


def test_inference(
    output_path: Path,
    model_path: Path = Path(__file__).parent / "models" / "L1-Qwen-1.5B-Exact",
    batch_size: int = 2
):
    print("Loading test questions...")
    test_df = load_questions(train=False)
    
    print("Loading regressor network...")
    network = Regressor.load_network()
    bins = network.bins
    
    print(f"Loading L1 model from {model_path}...")
    bundle = load_model_bundle(model_path)
    model, tokenizer, device = bundle.model, bundle.tokenizer, bundle.device
    
    # Folder stuff
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Process in batches
    num_batches = int(np.ceil(len(test_df) / batch_size))
    
    for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(test_df))
        batch_df = test_df.iloc[start_idx:end_idx]
        
        problems = batch_df["problem"].tolist()
        solutions = batch_df["solution"].tolist()
        question_ids = batch_df.index.tolist()
        
        # Step 1: Get initial hidden states (before "Think for X tokens")
        initial_hidden = get_initial_hidden_states(model, tokenizer, device, problems)
        
        # Step 2: Pass to regressor to get predicted target tokens
        predicted_targets, targets_p = numpy_wrapper_predict_batch(network, initial_hidden)
        
        # Step 3: Build full prompts with predicted targets
        full_prompts = [
            build_prompt(problem, target) 
            for problem, target in zip(problems, predicted_targets, strict=True)
        ]
        
        # Step 4: Run L1 inference with full prompts
        generated_texts, actual_token_counts = generate_with_prompt(
            model, tokenizer, device, full_prompts
        )
        
        all_results = []
        # Step 5: Evaluate and record results
        for qid, problem, solution, target, actual, gen_text, target_p in zip(
            question_ids, problems, solutions, predicted_targets, actual_token_counts, generated_texts, targets_p, strict=True
        ):
            is_correct = evaluate_answer(solution, gen_text)
            
            row = {
                "question_id": qid,
                "problem": problem,
                "expected_solution": solution,
                "target_tokens": target,
                "actual_tokens": actual,
                "is_correct": is_correct,
                "generated_text": gen_text,
            }

            for b, p in zip(bins, target_p, strict=True):
                row[f"bin_{b}"] = p
            
            all_results.append(row)
    
        # Save results to CSV
        results_df = pd.DataFrame(all_results)
        results_df.to_parquet(output_path / f"batch_{batch_idx}.parquet", index=False)
    
    return results_df

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--type", required=True, type=str,  choices=["static", "dynamic"])
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--every", type=int, default=50)

    return p.parse_args()

if __name__ == "__main__":
    # Run test inference

    args = parse_args()
    

    if args.type == "static":
        out_folder = Path(__file__).parent / "static_regressor_results"
        test_inference(
            out_folder,
            batch_size=args.batch
        )
    elif args.type == "dynamic":
        out_folder = Path(__file__).parent / "dynamic_regressor_results"
        test_inference_dynamic(
            out_folder,
            batch_size=args.batch,
            n_tokens=args.every
        )

    