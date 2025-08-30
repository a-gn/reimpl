import jax.numpy as jnp
from flax import nnx
from jax.typing import ArrayLike


class SimpleMLP(nnx.Module):
    def __init__(
        self, input_size: int, linear_layer_sizes: tuple[int, ...], rngs: nnx.Rngs
    ):
        linear_layer_sizes = (input_size,) + linear_layer_sizes
        self.linear_layers = [
            nnx.Linear(linear_layer_sizes[i - 1], linear_layer_sizes[i], rngs=rngs)
            for i in range(1, len(linear_layer_sizes))
        ]
        self.activation = nnx.relu

    def __call__(self, x: ArrayLike):
        x_array = jnp.array(x)
        for layer in self.linear_layers:
            x_array = self.activation(layer(x_array))
        return x_array


x = jnp.ones((3, 20))
rngs = nnx.Rngs(7)
network = SimpleMLP(20, (3, 4, 5), rngs=rngs)
y = network(x)
print(x)
print(y)
