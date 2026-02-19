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
    data_path = Path(__file__).parent.parent / "dataset_splitting"
    
    h_path = "/scratch/s3799042/results_nlp/test_hidden_states.npy" #data_path / "hidden_states_test.npy"
    h = jnp.load(h_path, allow_pickle= True).item()
    
    network = Regressor.load_network()
    
    df = pd.read_parquet(
        data_path / "test.parquet", 
        columns=['question_id', 'target_think_tokens']
    )

    #df = df.set_index("question_id")
    mask = np.isin(df['question_id'], h['ids'])
    df = df[mask]
    #df = df.reindex(h["ids"])
    emb = jnp.squeeze(h['embeddings'])
    bins = np.sort(df["target_think_tokens"].unique())
    
    batch_size = 1024
    num_samples = emb.shape[0]
    bucket_indices = []

    for i in range(0, num_samples, batch_size):
        batch_h = emb[i : i + batch_size]
        b_i, _ = predict_batch(network, batch_h)
        bucket_indices.append(b_i)

    bucket_indices = np.array(jnp.concatenate(bucket_indices))
    target_tokens = bins[bucket_indices]
    
    #np.save(Path(__file__).parent / "test_target_tokens.npy", target_tokens)
    np.save(
        Path(__file__).parent / "test_target_tokens.npy",
        {
            "ids": h['ids'],
            "target": target_tokens,
        },
    )

if __name__ == "__main__":
    main()