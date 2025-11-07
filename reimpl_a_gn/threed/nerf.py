import flax.nnx as nnx
import jax
import jax.numpy as jnp
import jax.typing as jt


class CoarseMLP(nnx.Module):
    """The coarse network that lets us choose interesting sampling positions in a NeRF."""

    def __init__(
        self,
        input_features: int,
        mid_features: tuple[int, ...],
        out_features: int,
        rngs: nnx.Rngs,
    ) -> None:
        input_size_per_layer = (input_features,) + mid_features
        output_size_per_layer = mid_features + (out_features,)
        self.linear_layers = nnx.List(
            [
                nnx.Linear(single_input_size, single_output_size, rngs=rngs)
                for single_input_size, single_output_size in zip(
                    input_size_per_layer, output_size_per_layer
                )
            ]
        )

    def __call__(
        self,
        rays: jt.ArrayLike,
    ):
        """Predict features for the given rays.

        @param rays Origins and direction unit vectors for all rays.
            Shape: `(number_of_rays, 6)`. Last axis: x, y, z, dx, dy, dz.
        @return Predicted features. Shapes: `(number_of_rays, self.out_features)`.

        """
        x = jnp.array(rays)
        for layer in self.linear_layers:
            x = nnx.relu(layer(x))
        return x


class FineMLP(nnx.Module):
    """The fine network that gives us more precise details in a NeRF."""

    def __init__(
        self,
        input_features: int,
        mid_features: tuple[int, ...],
        out_features: int,
        rngs: nnx.Rngs,
    ) -> None:
        input_size_per_layer = (input_features,) + mid_features
        output_size_per_layer = mid_features + (out_features,)
        self.linear_layers = nnx.List(
            [
                nnx.Linear(single_input_size, single_output_size, rngs=rngs)
                for single_input_size, single_output_size in zip(
                    input_size_per_layer, output_size_per_layer
                )
            ]
        )

    def __call__(
        self,
        rays: jt.ArrayLike,
    ) -> jax.Array:
        """Predict features for the given rays.

        @param rays Origins and direction unit vectors for all rays.
            Shape: `(number_of_rays, 6)`. Last axis: x, y, z, dx, dy, dz.
        @return Predicted features. Shapes: `(number_of_rays, self.out_features)`.

        """
        x = jnp.array(rays)
        for layer_index, layer in enumerate(self.linear_layers):
            x = layer(x)
            # all layers but last are ReLU-activated
            if layer_index <= len(self.linear_layers) - 2:
                x = nnx.relu(x)
        x = nnx.sigmoid(x)
        return x
