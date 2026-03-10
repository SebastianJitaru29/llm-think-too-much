import pandas as pd
import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path
from architecture import Regressor
from functools import partial

path_test_targets_hidden = Path(__file__).parent.parent / "data" / "processed" / "dataset_splitting"/ "test_targets_hidden_bert.parquet" #"eval_L1_bert" / "gsm8k_olympiad_amc_hidden_L1.parquet"

@partial(jax.jit, static_argnames=['predict_too_hard'])
def predict_batch(network: Regressor, hidden: jax.Array, predict_too_hard: bool = False) -> tuple[jax.Array, jax.Array]:
    logits = Regressor.forward(hidden, network)
    p = jax.nn.sigmoid(logits)
    bins_correct = p > 0.6
    
    any_correct = jnp.any(bins_correct, axis=1)
    highest_incorrect = jnp.argmax(p, axis=1)
    min_correct = jnp.argmax(bins_correct, axis=1)

    # Instead of highest 
    if predict_too_hard:
        bucket_i = jax.lax.select(any_correct, min_correct, jnp.full(shape=(750), fill_value=-1))
    else:
        bucket_i = jax.lax.select(any_correct, min_correct, highest_incorrect)

    return bucket_i, p

def main():
    #data_path = Path(__file__).parent.parent / "dataset_splitting"
    
    #h_path = "/scratch/s3799042/results_nlp/test_hidden_states.npy" #data_path / "hidden_states_test.npy"
    #h = jnp.load(h_path, allow_pickle= True).item()
    
    network = Regressor.load_network(name="regressor_bert.pkl")#"regressor_bert.pkl")#
    
    df = pd.read_parquet(path_test_targets_hidden)
    df = df.rename(columns={'id': 'question_id'})
    bins = np.linspace(100, 5000, num=20, dtype=int)
    #np.sort(df["target_think_tokens"].unique())

    df_unique = df.drop_duplicates(subset = 'question_id', keep = 'first')

    emb_array = np.stack(df_unique['hidden'].values)  # shape: (num_samples, embedding_dim)

    # Convert to JAX array
    emb = jnp.array(emb_array)

    
    batch_size = 1024
    num_samples = emb.shape[0]
    bucket_indices = []
    probs = []

    for i in range(0, num_samples, batch_size):
        batch_h = emb[i : i + batch_size]
        b_i, p_i = predict_batch(network, batch_h, predict_too_hard = False)
        bucket_indices.append(b_i)
        probs.append(p_i)

    bucket_indices = np.array(jnp.concatenate(bucket_indices))
    probs = np.array(jnp.concat(probs))
    target_tokens = bins[bucket_indices]

    
    #np.save(Path(__file__).parent / "test_target_tokens.npy", target_tokens)
    np.save(
        Path(__file__).parent / "results_bert_test.npy",
        {
            "ids": df_unique['question_id'],
            "target": target_tokens,
        },
    )

if __name__ == "__main__":
    main()