from typing import Protocol

import jax.numpy as jnp


class TrainableModule(Protocol):
    """Module with parameters that can be trained."""

    def __call__(self, *args, **kwargs) -> jnp.ndarray:
        """Forward pass of the module."""
        ...

    def get_trainable_parameters(self) -> list[jnp.ndarray]:
        """Get all of this module's trainable parameters."""
        ...
