import jax.numpy as jnp
from flax.linen import softmax
from jax import Array
from jax.typing import ArrayLike


def self_attention(data: ArrayLike, Q: ArrayLike, K: ArrayLike, V: ArrayLike) -> Array:
    """Simple self-attention layer."""

    data = jnp.asarray(data)
    Q = jnp.asarray(Q)
    K = jnp.asarray(K)
    V = jnp.asarray(V)

    if Q.shape[-1] != data.shape[-1]:
        raise ValueError(
            f"Q and data must have the same size along dot-product dimension, shapes are {Q.shape} and {data.shape}"
        )
    if K.shape[-1] != data.shape[-1]:
        raise ValueError(
            f"K and data must have the same size along dot-product dimension, shapes are {K.shape} and {data.shape}"
        )
    if V.shape[-1] != data.shape[-1]:
        raise ValueError(
            f"V and data must have the same size along dot-product dimension, shapes are {V.shape} and {data.shape}"
        )

    queries = Q @ data
    keys = V @ data
    values = V @ data

    attention_weights = softmax((queries @ keys) / jnp.sqrt(queries.shape[-1]))
    return attention_weights * values
