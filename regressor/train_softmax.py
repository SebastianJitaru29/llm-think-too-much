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

path_train_targets_hidden = Path(__file__).parent.parent / "data" / "processed" / "dataset_splitting" / "train_targets_hidden_bert.parquet"
MODEL_DIR = Path(__file__).parent.parent / "data" / "models" / "regressors"
MODEL_NAME = "L1_softmax.plt"

USE_L1_HIDDEN_STATES = False


def create_regressor_dataset(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:

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

    bins = np.sort(df["target_think_tokens"].unique())

    df_unique = df.drop_duplicates(subset=["question_id", "target_think_tokens"], keep="first")

    y_df = df_unique.pivot(index="question_id", columns="target_think_tokens", values="is_correct")
    y_df = y_df.fillna(0).astype(np.int32)

    hidden_df = df.drop_duplicates(subset=["question_id"]).set_index("question_id")
    hidden_df = hidden_df.loc[y_df.index]

    x = np.stack(hidden_df["hidden"].tolist())
    y = y_df.to_numpy()

    min_token_idx = np.argmax(y, axis=1)
    min_token_idx[~y.any(axis=1)] = len(bins)

    y = np.eye(len(bins) + 1, dtype=int)[min_token_idx]
    return x, y, tuple(bins)


def loss_dropout(
    network: Regressor,
    x: jax.Array,
    y: jax.Array,
    key: jax.Array,
    neg_scale: float = 2.5,
):
    yhat_logits = Regressor.forward_dropout(x, network, key)
    loss = optax.softmax_cross_entropy(yhat_logits, y)
    return loss.mean()


def loss_stats(
    network: Regressor,
    x: jax.Array,
    y: jax.Array,
):
    logits = Regressor.forward(x, network)

    loss = optax.softmax_cross_entropy(logits, y).mean()

    pred = jnp.argmax(logits, axis=1)
    true = jnp.argmax(y, axis=1)

    accuracy = (pred == true).mean()

    last_class = y.shape[1] - 1
    mask_last = (true == last_class)
    correct_last = (pred == true) & mask_last

    num_last = mask_last.sum()
    class_last_acc = jnp.where(
        num_last > 0,
        correct_last.sum() / num_last,
        1.0
    )

    return loss, accuracy, class_last_acc


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
):
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
    df = pd.read_parquet(path=path_train_targets_hidden)
    return create_regressor_dataset(df)


def compute_class_statistics(network, x, y):

    x_j = jax.device_put(x)
    y_j = jax.device_put(y)

    logits = Regressor.forward(x_j, network)

    pred = np.array(jnp.argmax(logits, axis=1))
    true = np.array(jnp.argmax(y_j, axis=1))

    n_classes = y.shape[1]

    stats = []
    for c in range(n_classes):

        mask = (true == c)
        count = mask.sum()

        if count > 0:
            acc = (pred[mask] == true[mask]).mean()
        else:
            acc = 1.0

        stats.append((c, count, acc))

    return stats


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
    val_batch_size = x.shape[0] - train_size

    train_idx, valid_idx = perm[:train_size], perm[train_size:]
    x_train, y_train = x[train_idx], y[train_idx]
    x_valid, y_valid = x[valid_idx], y[valid_idx]

    if USE_L1_HIDDEN_STATES:
        layers = (4096, 1024, 512, 256, 21)
    else:
        layers = (768, 1024, 512, 256, 21)

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

        for x_batch, y_batch in batch_dataset(x_valid, y_valid, batch=val_batch_size):

            x_batch = jax.device_put(x_batch)
            y_batch = jax.device_put(y_batch)

            l, a, tnr = valid_step(x_batch, y_batch, network)

            ba.append(a.mean().item())
            bl.append(l.mean().item())
            btnr.append(tnr.mean().item())

        va.append(np.mean(ba))
        vl.append(np.mean(bl))
        vtnr.append(np.mean(btnr))

    return network, tl, vl, va, vtnr, x_train, y_train, x_valid, y_valid


if __name__ == "__main__":

    from matplotlib import pyplot as plt

    x, y, bins = get_dataset()

    network, tl, vl, va, vtnr, x_train, y_train, x_valid, y_valid = train(
        x,
        y,
        bins=bins,
        epochs=80,
        batch_size=32,
        dropout=0.10
    )

    Regressor.save_network(network, name=MODEL_NAME, dir=MODEL_DIR)

    train_stats = compute_class_statistics(network, x_train, y_train)
    valid_stats = compute_class_statistics(network, x_valid, y_valid)

    print("\nClass statistics:")
    print("Class | Train count | Train acc | Val count | Val acc")

    for (c, train_count, train_acc), (_, val_count, val_acc) in zip(train_stats, valid_stats):
        print(f"{c:5d} | {train_count:11d} | {train_acc:9.4f} | {val_count:9d} | {val_acc:7.4f}")

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(va, label="validation accuracy", color="tab:blue")
    ax.plot(vtnr, color="tab:orange", label="Accuracy impossible")

    ax.legend()
    ax.set_xlabel("Epochs", fontsize=12)

    fig.savefig("./L1_loss3.png")