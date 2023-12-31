import timeit

import jax.numpy as jnp
from jax import grad, jit, random, vmap


def func(x):
    return jnp.prod(jnp.exp(x))


func_dx = grad(func)

x = random.normal(random.PRNGKey(0), (10,))
print(x)
print(func(x))
print(func_dx(x))
