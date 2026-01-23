import pandas as pd
import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path
from architecture import Regressor

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

def main():
    h_path = Path(__file__).parent / "innit_hidden_states"
    data_path = Path(__file__).parent.parent / "data"
    
    h_path = h_path / "hidden_states_test.npy"
    h = jnp.squeeze(jnp.load(h_path))
    
    network = Regressor.load_network()
    
    df_aime = pd.read_parquet(
        data_path / "train.parquet", 
        columns=['target_think_tokens']
    )
    bins = np.sort(df_aime["target_think_tokens"].unique())
    
    batch_size = 1024
    num_samples = h.shape[0]
    bucket_indices = []

    for i in range(0, num_samples, batch_size):
        batch_h = h[i : i + batch_size]
        b_i, _ = predict_batch(network, batch_h)
        bucket_indices.append(b_i)

    bucket_indices = np.array(jnp.concatenate(bucket_indices))
    target_tokens = bins[bucket_indices]
    
    np.save(data_path / "test_target_tokens.npy", target_tokens)

if __name__ == "__main__":
    main()