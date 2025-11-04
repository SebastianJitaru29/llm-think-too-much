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

def load_df() -> pd.DataFrame:

    dfs = []
    path =  (Path(".") / "dataset" / "hidden").resolve()
    for path in path.glob("*.csv"):
        dfs.append(pd.read_csv(path))

    df = pd.concat(dfs, ignore_index=True)
    
    return df


def create_regressor_dataset(h: np.ndarray, h_ids: np.ndarray, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:

    actual = df.groupby(["question_id", "target_think_tokens"])["token_step"].max().reset_index()
    q = df.drop_duplicates(["question_id", "target_think_tokens"], keep="first")
    q = q.drop(columns="token_step")
    q = pd.merge(q, actual, "left", ["question_id", "target_think_tokens"], validate="1:1")
    q = q.rename(columns={"token_step": "actual_think_tokens"})
    q = q.loc[q.index.isin(h_ids)]

    bins = np.sort(q["target_think_tokens"].unique())

    correct = q[["is_correct", "target_think_tokens", "question_id"]].copy()
    correct["is_correct"] = correct["is_correct"].astype(np.bool_)
    correct = pd.pivot(correct, index="question_id", columns="target_think_tokens", values="is_correct")
    correct = correct.astype(float).fillna(0)
    correct = correct.sort_index(axis="columns")

    return np.squeeze(h), correct.to_numpy().astype(np.int32), tuple(bins)

def loss_dropout(
    network: Regressor,
    x: jax.Array,
    y: jax.Array,
    key: jax.Array
):
    yhat = Regressor.forward_dropout(x, network, key)
    return optax.sigmoid_binary_cross_entropy(yhat, y).mean()

def loss_accuracy(
    network: Regressor,
    x: jax.Array,
    y: jax.Array,
):
    yhat_logits = Regressor.forward(x, network)
    yhat = jax.nn.sigmoid(yhat_logits)
    labelhat = (yhat > 0.5).astype(jnp.int32)
    accuracy = (labelhat == y).astype(jnp.float32).mean(axis=1)

    return optax.sigmoid_binary_cross_entropy(yhat_logits, y).mean(), accuracy.mean()

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
    l, a = loss_accuracy(network, x, y)
    return l, a


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
    df = load_df()
    h = np.load("./innit_hidden_states/hidden_states.npy").astype(np.float32)
    h_ids = np.arange(h.shape[0])

    if not train:
        n = h_ids.shape[0]
        take_mask = np.zeros(n, dtype=np.bool)
        take_mask[np.random.choice(n, size=10, replace=False)] = True

        h = h[take_mask]
        h_ids = h_ids[take_mask]

    return create_regressor_dataset(h, h_ids, df)



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

    
    layers = (1_536, 256, 256, 256, 10)

    network = Regressor(
        arch=Regressor.init_mlp(
            layers=layers,
            key=kinit
        ),
        bins=bins,
        layers=layers,
        dropout=dropout
    )

    optimizer = optax.adamw(1e-3, weight_decay=1e-5)
    opt_state = optimizer.init(network)

    train_step_compiled = jax.jit(partial(train_step, optimizer=optimizer))

    tl = []
    vl = []
    va = []
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
        for x_batch, y_batch in batch_dataset(x_valid, y_valid, batch=batch_size):

            x_batch = jax.device_put(x_batch)
            y_batch = jax.device_put(y_batch)

            l, a = valid_step(x_batch, y_batch, network)

            ba.append(a.mean().item())
            bl.append(l.mean().item())

        va.append(np.mean(ba))
        vl.append(np.mean(bl))
    
    return network, tl, vl, va


def calc_baseline_accuracy(y: np.ndarray) -> float:
    c = np.unique_counts(y).counts
    return c.max() / c.sum()

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    import matplotlib
    matplotlib.use("Qt5Agg")

    x, y, bins = get_dataset()
    network, tl, vl, va = train(x, y, bins=bins, epochs=100, batch_size=512, dropout=0.2)

    Regressor.save_network(network)

    basline_acc = calc_baseline_accuracy(y)

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    l1, = ax2.plot(tl, label="train loss", color="tab:blue")
    l2, = ax2.plot(vl, label="valid loss", color="tab:orange")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")

    l3, = ax1.plot(va, label="valid acc", color="tab:green", linestyle="dashed")
    l4 = ax1.axhline(basline_acc, color="black", linestyle="dashed", label="basline acc")
    ax1.set_ylabel("Accuracy")

    lines = [l1, l2, l3, l4]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="center right")

    plt.show()

