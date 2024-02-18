from abc import ABC, abstractmethod

import jax.numpy as jnp


class TrainableModule(ABC):
    """Module with parameters that can be trained."""

    @abstractmethod
    def get_trainable_parameters(self) -> list[jnp.ndarray]:
        """Get all of this module's trainable parameters."""
        pass
