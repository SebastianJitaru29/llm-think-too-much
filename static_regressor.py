
from pathlib import Path
import jax
import jax.numpy as jnp
import pandas as pd
import torch

from regressor.architecture import Regressor

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
    
def test_inference(
    batch_size: int = 2
):

    test_df = load_questions(train=False)
    network = Regressor.load_network()

    # Load the L1 model

    # Get the initial hidden states (with the question loaded but before "think for ....")

    # Pass the initial hidden state to the regressor numpy_wrapper_predict_batch and get the target tokens

    # Add the "think for.."" phrase with build_prompt using the target tokens

    # Run the l1 inference 
     
    # For each batch create csv datatset with question_id, actual_tokens (the actual amount generated), target_tokens, correct, gen_output_text (with special tokens)]

    
    
    