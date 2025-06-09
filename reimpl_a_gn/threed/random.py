"""Utilities for random sampling."""

import jax
import jax.numpy as jnp
import jax.typing as jt


def piecewise_uniform(
    key: jt.ArrayLike,
    intervals: jt.ArrayLike,
    pdf_values: jt.ArrayLike,
    sample_count_per_distribution: int,
):
    """Sample from a piecewise uniform distribution.

    @param key The PRNG key to use for `jax.random`.
    @param distributions Piecewise-uniform distributions' interval bounds. Those are the "pieces" that internally have
    uniform probabilities.
    Shape: `(distribution_count, interval_count + 1)`.
    @param pdf_values Piecewise-uniform distributions' probability density values. One for every interval.
    The sum of the probabilities weighted by the interval lengths must be one.
    Shape: `(distribution_count, interval_count)`.
    @param sample_count_per_distribution The number of values to sample from each distribution.
    @return The values sampled from all distribution. Shape: `(distribution_count, sample_count_per_distribution)`.

    """

    intervals = jnp.array(intervals)
    pdf_values = jnp.array(pdf_values)

    if intervals.ndim != 2:
        raise ValueError(
            f"intervals should have shape (N, interval_count + 1), got shape {intervals.shape}"
        )
    if intervals.shape[1] < 2:
        raise ValueError(
            f"intervals should have a second axis of size at least 2 to define an interval"
            f", but it has shape {intervals.shape}"
        )
    expected_pdf_values_shape = (intervals.shape[0], intervals.shape[1] - 1)
    if pdf_values.ndim != 2 or pdf_values.shape != expected_pdf_values_shape:
        raise ValueError(
            f"pdf_values should have shape {expected_pdf_values_shape} based on intervals' shape {intervals.shape}"
            f", but it has shape {pdf_values.shape} instead"
        )

    total_bounds = jnp.stack([intervals[:, 0], intervals[:, 1]], axis=-1)
    assert total_bounds.shape == (intervals.shape[0], 2)

    # sample uniform values within the total bounds
    uniform_values = jax.random.uniform(
        key,
        (intervals.shape[0], sample_count_per_distribution),
        dtype=float,
        minval=total_bounds[:, 0],
        maxval=total_bounds[:, 1],
    )

    # scale these values based on the sizes and probability densities of intervals
