from typing import Callable, Protocol

import jax.numpy as jnp


class TrainableModule(Protocol):
    """Module with parameters that can be trained."""

    __call__: Callable[..., jnp.ndarray]

    def get_trainable_parameters(self) -> list[jnp.ndarray]:
        """Get all of this module's trainable parameters."""
        ...
