import pandas as pd
import numpy as np
from pathlib import Path
from typing import Generator
import jax
import optax
import jax.numpy as jnp
import tqdm
from functools import partial

from architecture import Regressor

path_train_targets_hidden = Path(__file__).parent.parent / "data" / "processed" / "dataset_splitting" / "train_targets_hidden.parquet"

def create_regressor_dataset(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    
    # 1. Clean up 'is_correct' to boolean
    if df["is_correct"].dtype == object:
        df["is_correct"] = (
            df["is_correct"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({"true": True, "false": False})
        )
    else:
        df["is_correct"] = df["is_correct"].astype(bool)

    # 2. Determine the unique bins from target_think_tokens
    bins = np.sort(df["target_think_tokens"].unique())

    # 3. Deduplicate to ensure one unique (question_id, token_count) pair per row
    df_unique = df.drop_duplicates(subset=["question_id", "target_think_tokens"], keep="first")

    # 4. Pivot to create the y matrix (targets)
    y_df = df_unique.pivot(index="question_id", columns="target_think_tokens", values="is_correct")
    y_df = y_df.fillna(0).astype(np.int32)
    
    # 5. Extract one hidden state per question_id
    # We can use the original df here, just grabbing the first occurrence of each question
    hidden_df = df.drop_duplicates(subset=["question_id"]).set_index("question_id")
    
    # 6. Align the hidden states to the exact index order of y_df
    hidden_df = hidden_df.loc[y_df.index]
    
    # 7. Stack the hidden state arrays into a 2D numpy array for x
    x = np.stack(hidden_df["hidden"].tolist())
    y = y_df.to_numpy()
    
    return x, y, tuple(bins)



def loss_dropout(
    network: Regressor,
    x: jax.Array,
    y: jax.Array,
    key: jax.Array,
    neg_scale: float = 2.5,
):
    yhat_logits = Regressor.forward_dropout(x, network, key)
    loss = optax.sigmoid_binary_cross_entropy(yhat_logits, y)
    weights = jnp.where(y, 1.0, neg_scale)
    return (loss * weights).mean()


def loss_stats(
    network: Regressor,
    x: jax.Array,
    y: jax.Array,
    neg_scale: float = 2.5,
):
    
    label = (y > 0.5)

    yhat_logits = Regressor.forward(x, network)
    yhat = jax.nn.sigmoid(yhat_logits)
    labelhat = (yhat > 0.5).astype(jnp.int32)
    accuracy = (labelhat == label).astype(jnp.float32).mean(axis=1)

    negatives = (~label).sum()
    tn = ((~label) & (~labelhat)).sum()
    tnr = tn / negatives

    loss = optax.sigmoid_binary_cross_entropy(yhat_logits, y)
    weights = jnp.where(y, 1.0, neg_scale)
    weighted_loss = (loss * weights).mean()

    return weighted_loss, accuracy.mean(), tnr


def train_step(
    x: jax.Array,
    y: jax.Array,
    key: jax.Array,
    network: Regressor,
    opt_state: jax.Array,
    optimizer: optax.GradientTransformationExtraArgs,
) -> tuple[jax.Array, Regressor, jax.Array]:

    l, grad = jax.value_and_grad(loss_dropout)(network, x, y, key)
    updates, opt_state = optimizer.update(grad, opt_state, network)
    network = optax.apply_updates(network, updates)

    return l, network, opt_state
    
@jax.jit
def valid_step(
    x: jax.Array,
    y: jax.Array,
    network: Regressor,
) -> tuple[jax.Array, Regressor, jax.Array]:
    return loss_stats(network, x, y)


def batch_dataset(x: np.ndarray, y: np.ndarray, batch: int, shuffle: bool = False) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
    
    n = x.shape[0]

    if shuffle:
        si = np.random.permutation(n)
        x = x[si, :]
        y = y[si, :]

    n_batches = int(np.ceil(n / batch)) 
    for i_batch in range(n_batches):

        s = i_batch * batch
        e = s + batch

        yield x[s:e, :], y[s:e, :]

    
def get_dataset(train: bool = True) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    df = pd.read_parquet(path = path_train_targets_hidden)

    return create_regressor_dataset(df)



def train(
    x: np.ndarray,
    y: np.ndarray,
    epochs: int,
    bins: tuple[int, ...],
    dropout: float = 0.3,
    valid_p: float = 0.1,
    batch_size: int = 256,
):
    
    key = jax.random.key(seed=0)
    kinit, kloop, kvalid, k4, k5, key = jax.random.split(key, 6)
    
    perm = np.random.permutation(x.shape[0])
    train_size = int((1 - valid_p) * x.shape[0])
    train_idx, valid_idx = perm[:train_size], perm[train_size:]
    x_train, y_train = x[train_idx], y[train_idx]
    x_valid, y_valid = x[valid_idx], y[valid_idx]

    
    layers = (4_096, 1024, 512, 256, 20)

    network = Regressor(
        arch=Regressor.init_mlp(
            layers=layers,
            key=kinit
        ),
        bins=bins,
        layers=layers,
        dropout=dropout
    )

    optimizer = optax.adamw(1e-4, weight_decay=1e-4)
    opt_state = optimizer.init(network)

    train_step_compiled = jax.jit(partial(train_step, optimizer=optimizer))

    tl = []
    vl = []
    va = []
    vtnr = []
    for _ in tqdm.tqdm(range(epochs)):

        bl = []
        for x_batch, y_batch in batch_dataset(x_train, y_train, batch=batch_size, shuffle=True):
            x_batch = jax.device_put(x_batch)
            y_batch = jax.device_put(y_batch)

            k_use_loop, kloop = jax.random.split(kloop, 2)

            l, network, opt_state = train_step_compiled(x_batch, y_batch, k_use_loop, network, opt_state)
            bl.append(l.mean().item())

        tl.append(np.mean(bl))
        
        bl = []
        ba = []
        btnr = []
        for x_batch, y_batch in batch_dataset(x_valid, y_valid, batch=batch_size):

            x_batch = jax.device_put(x_batch)
            y_batch = jax.device_put(y_batch)

            l, a, tnr = valid_step(x_batch, y_batch, network)

            ba.append(a.mean().item())
            bl.append(l.mean().item())
            btnr.append(tnr.mean().item())

        va.append(np.mean(ba))
        vl.append(np.mean(bl))
        vtnr.append(np.mean(btnr))
    
    return network, tl, vl, va, vtnr



def calc_baseline_stats(y: np.ndarray) -> tuple[float, float]:
    c = np.unique_counts(y).counts
    ba = c.max() / c.sum()
    
    negatives = (y == 0).sum()
    negatives_p = negatives / y.size
    
    negatives_p = (y == 0).mean()
    tnr = negatives_p  # expected TNR under random prediction using data
    
    return ba, tnr
     

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    # import matplotlib
    # matplotlib.use("Qt5Agg")

    x, y, bins = get_dataset()
    network, tl, vl, va, vtnr = train(x, y, bins=bins, epochs=140, batch_size=512, dropout=0.20)

    Regressor.save_network(network, name = "regressor_correct_aime.pkl")

    basline_acc, baseline_tnr = calc_baseline_stats(y)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(va, label="validation accuracy", color="tab:blue")
    ax.axhline(basline_acc, color="tab:blue", linestyle="dashed", label="basline acccuracy")

    ax.axhline(baseline_tnr, color="tab:orange", linestyle="dashed", label="basline TNR")
    ax.plot(vtnr, color="tab:orange", label="validation TNR")
    ax.legend()

    ax.set_xlabel("Epochs", fontsize=12)

    fig.savefig("./perf2.png")

