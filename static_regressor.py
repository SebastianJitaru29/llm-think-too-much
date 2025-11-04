
from pathlib import Path
import os
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

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
def predict_batch(network: Regressor, hidden: jax.Array) -> jax.Array:
    logits = Regressor.forward(hidden, network)
    p = jax.nn.sigmoid(logits)
    bins_correct = p > 0.5
    
    any_correct = jnp.any(bins_correct, axis=1)
    highest_incorrect = jnp.argmax(p, axis=1)
    min_correct = jnp.argmax(bins_correct, axis=1)

    bucket_i = jax.lax.select(any_correct, min_correct, highest_incorrect)

    return bucket_i


def numpy_wrapper_predict_batch(network: Regressor, hidden: torch.Tensor) -> list[int]:
    
    hidden = jax.device_put(hidden)
    bucket_i = predict_batch(network, hidden)
    target_tokens = []

    for bi in bucket_i:
        target_tokens.append(network.bins[bi])
    
    assert len(target_tokens) == hidden.shape[0]

    return target_tokens

def load_questions(
    train: bool = True
):
    df = pd.read_parquet(Path(__file__).parent / "data" / "dataset.parquet")

    if not train:
        return df.sample(10)
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

def test_inference(
    model_path: str = "./models/L1-Qwen-1.5B-Exact",
    output_path: str = "./regressor_test_results.csv",
    batch_size: int = 2
):
    """
    Test inference using the regressor to predict optimal thinking tokens.
    
    Args:
        model_path: Path to the L1 model
        output_path: Path to save results CSV
        batch_size: Batch size for inference
        num_samples: Number of test samples to use
    """
    print("Loading test questions...")
    test_df = load_questions(train=False)
    
    print("Loading regressor network...")
    network = Regressor.load_network()
    
    print(f"Loading L1 model from {model_path}...")
    bundle = load_model_bundle(model_path)
    model, tokenizer, device = bundle.model, bundle.tokenizer, bundle.device
    
    all_results = []
    
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
        predicted_targets = numpy_wrapper_predict_batch(network, initial_hidden)
        
        # Step 3: Build full prompts with predicted targets
        full_prompts = [
            build_prompt(problem, target) 
            for problem, target in zip(problems, predicted_targets, strict=True)
        ]
        
        # Step 4: Run L1 inference with full prompts
        generated_texts, actual_token_counts = generate_with_prompt(
            model, tokenizer, device, full_prompts
        )
        
        # Step 5: Evaluate and record results
        for qid, problem, solution, target, actual, gen_text in zip(
            question_ids, problems, solutions, predicted_targets, actual_token_counts, generated_texts
        ):
            is_correct = evaluate_answer(solution, gen_text)
            
            all_results.append({
                "question_id": qid,
                "problem": problem,
                "expected_solution": solution,
                "target_tokens": target,
                "actual_tokens": actual,
                "is_correct": is_correct,
                "generated_text": gen_text,
            })
    
    # Save results to CSV
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_path, index=False)
    
    # Print summary statistics
    print(f"Total samples: {len(results_df)}")
    print(f"Correct answers: {results_df['is_correct'].sum()} ({results_df['is_correct'].mean()*100:.1f}%)")
    print(f"Mean target tokens: {results_df['target_tokens'].mean():.1f}")
    print(f"Mean actual tokens: {results_df['actual_tokens'].mean():.1f}")
    print(f"Mean token difference: {(results_df['actual_tokens'] - results_df['target_tokens']).abs().mean():.1f}")
    print(f"\nResults saved to: {output_path}")
    
    return results_df

if __name__ == "__main__":
    # Run test inference
    results = test_inference(
        model_path="./models/L1-Qwen-1.5B-Exact",
        output_path="./test_inference_results.csv",
        batch_size=2
    )
    