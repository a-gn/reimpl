"""Utilities for random sampling."""

import jax
import jax.numpy as jnp
import jax.typing as jt


def piecewise_uniform(
    key: jt.ArrayLike,
    intervals: jt.ArrayLike,
    pdf_values: jt.ArrayLike,
    sample_count_per_distribution: int,
) -> jax.Array:
    """Sample from a piecewise uniform distribution.

    @param key The PRNG key to use for `jax.random`.
    @param distributions Piecewise-uniform distributions' interval bounds. Those are the "pieces" that internally have
    uniform probabilities.
    Shape: `(..., interval_count + 1)`.
    @param pdf_values Piecewise-uniform distributions' probability density values. One for every interval.
    The sum of the probabilities weighted by the interval lengths must be one.
    Shape: `(..., interval_count)`.
    @param sample_count_per_distribution The number of values to sample from each distribution.
    @return The values sampled from all distribution. Shape: `(..., sample_count_per_distribution)`.

    """

    intervals = jnp.array(intervals)
    pdf_values = jnp.array(pdf_values)

    interval_lengths = intervals[..., 1:] - intervals[..., :-1]
    interval_probas = interval_lengths * pdf_values

    interval_choice_key, key = jax.random.split(key)
    chosen_intervals = jax.random.categorical(
        key=interval_choice_key,
        # add broadcast axis for multiple samples per distribution
        logits=jnp.expand_dims(jnp.log(interval_probas), -1),
        axis=-2,  # category axis is followed by broadcast axis
        shape=intervals.shape[:-1] + (sample_count_per_distribution,),
    )
    del interval_choice_key

    chosen_interval_lower_bounds = jnp.take_along_axis(
        arr=intervals, indices=chosen_intervals, axis=-1
    )
    chosen_interval_upper_bounds = jnp.take_along_axis(
        arr=intervals, indices=chosen_intervals + 1, axis=-1
    )
    position_choice_key, key = jax.random.split(key)
    chosen_positions = jax.random.uniform(
        key=position_choice_key,
        shape=chosen_intervals.shape,
        minval=chosen_interval_lower_bounds,
        maxval=chosen_interval_upper_bounds,
    )
    del position_choice_key

    return chosen_positions
