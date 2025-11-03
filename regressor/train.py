import pandas as pd
import numpy as np
from pathlib import Path
from typing import Generator
import jax
import optax
import jax.numpy as jnp
import flax
import tqdm
from functools import partial

def load_df() -> pd.DataFrame:

    dfs = []
    path =  (Path(".") / "dataset" / "hidden").resolve()
    for path in path.glob("*.csv"):
        dfs.append(pd.read_csv(path))

    return pd.concat(dfs, ignore_index=True)

def create_regressor_dataset(h: np.ndarray, h_ids: np.ndarray, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:

    actual = df.groupby(["question_id", "target_think_tokens"])["token_step"].max().reset_index()
    q = df.drop_duplicates(["question_id", "target_think_tokens"], keep="first")
    q = q.drop(columns="token_step")
    q = pd.merge(q, actual, "left", ["question_id", "target_think_tokens"], validate="1:1")
    q = q.rename(columns={"token_step": "actual_think_tokens"})

    correct = q[["is_correct", "target_think_tokens", "question_id"]].copy()
    correct["is_correct"] = correct["is_correct"].astype(np.bool_)
    correct = pd.pivot(correct, index="question_id", columns="target_think_tokens", values="is_correct")
    correct = correct.astype(float).fillna(0)
    correct = correct.sort_index()

    q = q.loc[q.index.isin(h_ids)]

    return np.squeeze(h), correct.to_numpy()

@flax.struct.dataclass
class Network:
    layers: tuple[tuple[jax.Array, jax.Array]]


def forward(x: jax.Array, network: Network) -> jax.Array:
    
    h = x
    for w, b in network.layers[:-1]:
        h = h @ w.T + b 
        h = jax.nn.relu(h)
    
    w, b = network.layers[-1]
    logits = h @ w.T + b 
    return logits


def loss(
    network: Network,
    x: jax.Array,
    y: jax.Array,
):
    yhat = forward(x, network)
    return optax.sigmoid_binary_cross_entropy(yhat, y).mean()

def train_step(
    x: jax.Array,
    y: jax.Array,
    network: Network,
    opt_state: jax.Array,
    optimizer: optax.GradientTransformationExtraArgs
) -> tuple[jax.Array, Network, jax.Array]:

    l, grad = jax.value_and_grad(loss)(network, x, y)
    updates, opt_state = optimizer.update(grad, opt_state, network)
    network = optax.apply_updates(network, updates)

    return l, network, opt_state
    
@jax.jit
def valid_step(
    x: jax.Array,
    y: jax.Array,
    network: Network,
) -> tuple[jax.Array, Network, jax.Array]:
    l = loss(network, x, y)
    return l


def init_mlp(layers: list[int], key: jax.Array) -> tuple[tuple[jax.Array, jax.Array]]:

    arch = []

    initializer = jax.nn.initializers.he_normal()
    keys = jax.random.split(key, len(layers) - 1)
    
    for prev, fol, k in zip(layers[:-1], layers[1:], keys, strict=True):

        wkey, bkey = jax.random.split(k, 2)

        w = initializer(wkey, (fol, prev))
        b = jax.random.normal(bkey, shape=(fol,)) * (1e-3)

        arch.append((w, b))

    return tuple(arch)


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



def get_dataset() -> tuple[np.ndarray, np.ndarray]:
    df = load_df()
    h = np.load("./innit_hidden_states/hidden_states.npy").astype(np.float32)
    h_ids = np.arange(h.shape[0])
    return create_regressor_dataset(h, h_ids, df)



def train(
    x: np.ndarray,
    y: np.ndarray,
    epochs: int,
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

    
    network = Network(init_mlp(
        layers=[1_536, 256, 256, 256, 10],
        key=kinit
    ))

    optimizer = optax.adamw(1e-4, weight_decay=1e-3)
    opt_state = optimizer.init(network)

    train_step_compiled = jax.jit(partial(train_step, optimizer=optimizer))

    tl = []
    vl = []
    for _ in tqdm.tqdm(range(epochs)):

        k_use_loop, kloop = jax.random.split(kloop, 2) 

        bl = []
        for x_batch, y_batch in batch_dataset(x_train, y_train, batch=batch_size, shuffle=True):
            x_batch = jax.device_put(x_batch)
            y_batch = jax.device_put(y_batch)

            l, network, opt_state = train_step_compiled(x_batch, y_batch, network, opt_state)
            bl.append(l.mean().item())

        tl.append(np.mean(bl))
        
        bl = []
        for x_batch, y_batch in batch_dataset(x_valid, y_valid, batch=batch_size):

            x_batch = jax.device_put(x_batch)
            y_batch = jax.device_put(y_batch)

            l = valid_step(x_batch, y_batch, network)

            bl.append(l.mean().item())
        vl.append(np.mean(bl))
    
    return tl, vl



if __name__ == "__main__":
    from matplotlib import pyplot as plt

    x, y = get_dataset()
    tl, vl = train(x, y, epochs=30, batch_size=512)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(tl, label="train")
    ax.plot(vl, label="valid")

    ax.legend()
    plt.show()