from __future__ import annotations
from pathlib import Path

import jax
import jax.numpy as jnp
import flax
import pickle


class Regressor(flax.struct.PyTreeNode):
    
    
    arch: tuple[tuple[jax.Array, jax.Array]]
    layers: tuple[int, ...] = flax.struct.field(pytree_node=False)
    bins: tuple[int, ...] = flax.struct.field(pytree_node=False)
    dropout: float = flax.struct.field(pytree_node=False, default=0.3)

    
    @staticmethod
    def save_network(network: Regressor, name: str = "regressor.pkl", dir = Path(__file__).parent):
        dir.mkdir(parents=True, exist_ok=True)
        
        leaves, _ = jax.tree.flatten(network.arch)
        python_leaves = [x.tolist() for x in leaves]

        save_obj = {
            "layers": network.layers,
            "arch_flatten": python_leaves,
            "dropout": network.dropout,
            "bins": network.bins
        }

        with open(dir / name, "wb") as f:
            pickle.dump(save_obj, f)
    

    @classmethod
    def load_network(cls, name: str = "regressor.pkl") -> Regressor:
        
        with open(Path(__file__).parent / name, "rb") as f:
            load_obj = pickle.load(f)

        python_leaves = load_obj.pop("arch_flatten")
        leaves = [jnp.array(x) for x in python_leaves]

        degenerative_key = jax.random.key(seed=0)
        template = Regressor.init_mlp(load_obj["layers"], degenerative_key)
        _, treedef = jax.tree.flatten(template)

        load_obj["arch"] = jax.tree.unflatten(treedef, leaves)
        
        return cls(**load_obj)
        


    @staticmethod
    def forward_dropout(x: jax.Array, network: Regressor, key: jax.Array) -> jax.Array:

        n_iters = len(network.arch) - 1
        keys = jax.random.split(key, n_iters)
        p = network.dropout

        h = x
        for i in range(n_iters):

            w, b = network.arch[i]
            k = keys[i]
            h = h @ w.T + b 

            mask = jax.random.bernoulli(k, 1 - p, h.shape).astype(jnp.float32)
            h = jax.nn.relu(h)
            h = (h * mask) / (1 - p)
        
        w, b = network.arch[-1]
        logits = h @ w.T + b 
        return logits

    @staticmethod
    def forward(x: jax.Array, network: Regressor) -> jax.Array:
        
        n_iters = len(network.arch) - 1

        h = x
        for i in range(n_iters):

            w, b = network.arch[i]
            h = h @ w.T + b 
            h = jax.nn.relu(h)
        
        w, b = network.arch[-1]
        logits = h @ w.T + b 
        return logits

    @staticmethod
    def init_mlp(layers: tuple[int, ...], key: jax.Array) -> tuple[tuple[jax.Array, jax.Array]]:

        arch = []

        initializer = jax.nn.initializers.he_normal()
        keys = jax.random.split(key, len(layers) - 1)
        
        for prev, fol, k in zip(layers[:-1], layers[1:], keys, strict=True):

            wkey, bkey = jax.random.split(k, 2)

            w = initializer(wkey, (fol, prev))
            b = jax.random.normal(bkey, shape=(fol,)) * (1e-3)

            arch.append((w, b))

        return tuple(arch)
