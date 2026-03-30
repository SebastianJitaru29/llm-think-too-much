"""Generate regressor target tokens for eval_data.parquet.

Extracts hidden states from the LLM and runs the trained regressor
to predict optimal token budgets. Output .npy is compatible with
eval_pipeline.py --target-tokens.

Usage:
    python evaluation/generate_targets.py \
        --model-path /scratch/s6019595/models/L1-Qwen3-8B-Max/ \
        --regressor-path data/models/regressors/regressor_L1.pkl \
        --datasets math-500 aime-250 \
        --output data/processed/regressor_target_tokens/targets_eval.npy
"""

import argparse
import sys
from pathlib import Path
from functools import partial

import numpy as np
import pandas as pd
import torch
import tqdm
import jax
import jax.numpy as jnp

from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from regressor.architecture import Regressor

EVAL_DATASET_PATH = Path(__file__).parent.parent.parent / "data" / "raw" / "eval_data.parquet"


@partial(jax.jit, static_argnames=["predict_too_hard"])
def predict_batch(network: Regressor, hidden: jax.Array, predict_too_hard: bool = False) -> tuple[jax.Array, jax.Array]:
    logits = Regressor.forward(hidden, network)
    p = jax.nn.sigmoid(logits)
    bins_correct = p > 0.8

    any_correct = jnp.any(bins_correct, axis=1)
    highest_incorrect = jnp.argmax(p, axis=1)
    min_correct = jnp.argmax(bins_correct, axis=1)

    if predict_too_hard:
        bucket_i = jax.lax.select(any_correct, min_correct, jnp.full(shape=hidden.shape[0], fill_value=-1))
    else:
        bucket_i = jax.lax.select(any_correct, min_correct, highest_incorrect)

    return bucket_i, p


def extract_hidden_states(problems, model_path, batch_size=8):
    """Extract last-token hidden states from the LLM."""
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True,
    )
    model.eval()

    hidden_states = []
    for i in tqdm.tqdm(range(0, len(problems), batch_size), desc="Extracting hidden states"):
        batch = problems[i : i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=False, add_special_tokens=True)
        input_ids = enc["input_ids"].to(model.device)
        attention_mask = enc["attention_mask"].to(model.device)

        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, use_cache=False)

        batch_last_tokens = out.hidden_states[-1][:, -1, :]
        hidden_states.append(batch_last_tokens.cpu().numpy().astype(np.float16))

    del model
    torch.cuda.empty_cache()

    return np.concatenate(hidden_states, axis=0)


def main():
    parser = argparse.ArgumentParser(description="Generate regressor target tokens for eval data")
    parser.add_argument("--model-path", type=str, required=True, help="LLM to extract hidden states from")
    parser.add_argument("--regressor-path", type=str, required=True, help="Path to trained regressor .pkl")
    parser.add_argument("--datasets", type=str, nargs="+", default=["math-500", "gsm8k", "olympiad"])
    parser.add_argument("--output", type=str, required=True, help="Output .npy path")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--predict-too-hard", action="store_true",
                        help="Mark problems as -1 if no bin passes threshold")
    args = parser.parse_args()

    # Load eval data
    df = pd.read_parquet(EVAL_DATASET_PATH)
    df = df[df["dataset"].isin(args.datasets)]
    df = df.drop_duplicates(subset="id", keep="first").reset_index(drop=True)
    print(f"Loaded {len(df)} problems: {df['dataset'].value_counts().to_dict()}")

    # Extract hidden states
    problems = df["problem"].astype(str).tolist()
    hidden_states = extract_hidden_states(problems, args.model_path, args.batch_size)
    print(f"Hidden states shape: {hidden_states.shape}")

    # Load regressor and predict
    regressor_path = Path(args.regressor_path)
    network = Regressor.load_network(name=regressor_path.name, dir=regressor_path.parent)
    bins = np.linspace(100, 5000, num=20, dtype=int)

    emb = jnp.array(hidden_states)
    batch_size = 1024
    bucket_indices = []
    probs = []

    for i in range(0, emb.shape[0], batch_size):
        batch_h = emb[i : i + batch_size]
        b_i, p_i = predict_batch(network, batch_h, predict_too_hard=args.predict_too_hard)
        bucket_indices.append(b_i)
        probs.append(p_i)

    bucket_indices = np.array(jnp.concatenate(bucket_indices))
    probs = np.array(jnp.concatenate(probs))
    target_tokens = bins[bucket_indices]

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ids = df["id"].values
    np.save(output_path, {"ids": ids, "target": target_tokens})
    print(f"Saved targets for {len(ids)} problems to {output_path}")
    print(f"Target tokens — min={target_tokens.min()}, max={target_tokens.max()}, mean={target_tokens.mean():.0f}")


if __name__ == "__main__":
    main()
