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
    if jnp.any(pdf_values < 0):
        strictly_negative_pdf_value_count = jnp.count_nonzero(pdf_values < 0)
        raise ValueError(
            f"probability density values can't be strictly negative, but {strictly_negative_pdf_value_count} are"
        )

    interval_lengths = intervals[:, 1:] - intervals[:, :-1]
    assert interval_lengths.shape == pdf_values.shape
    # compute the probability of each uniform interval in the entire distributions
    interval_probabilities = interval_lengths * pdf_values
    # check that the universe in each distribution has probability 1.0
    total_sums_of_pdfs = jnp.sum(interval_probabilities, axis=1)
    if not jnp.allclose(total_sums_of_pdfs, 1.0):
        index_of_farthest_value_from_1 = jnp.argmax(
            jnp.abs(1.0 - jnp.sum(interval_probabilities, axis=1))
        )
        farthest_value_from_1 = total_sums_of_pdfs[index_of_farthest_value_from_1]
        raise ValueError(
            "the sum of interval probabilities should equal one, but doesn't"
            f": the farthest sum of probabilities in a distribution is {farthest_value_from_1}"
        )

    total_bounds = jnp.stack([intervals[:, 0], intervals[:, 1]], axis=-1)
    assert total_bounds.shape == (intervals.shape[0], 2)

    # sample uniform values within [0, 1], which we will use as positions within the input bounds weighted by PDF values
    position_sampling_key, key = jax.random.split(key)
    uniformly_sampled_positions = jax.random.uniform(
        key,
        (intervals.shape[0], sample_count_per_distribution),
        dtype=float,
        minval=0.0,
        maxval=1.0,
    )
    del position_sampling_key

    # map the uniform values to the piecewise-uniform intervals we want
    cumulative_interval_probabilities = jnp.cumulative_sum(
        interval_probabilities, axis=1
    )
    # find which interval each uniformly-sampled value belongs to
    final_sampled_values = jnp.zeros_like(
        intervals, shape=(intervals.shape[0], sample_count_per_distribution)
    )
    for distribution_index in range(pdf_values.shape[0]):
        # find where [0,1] positions fall within the cumulative sum of interval probabilities
        uniform_value_to_interval = jnp.searchsorted(
            cumulative_interval_probabilities[distribution_index, :],
            uniformly_sampled_positions[distribution_index, :],
            side="right",  # so we're always one above the interval's lower bound's position
        )
        # index the final values and store them
        final_sampled_values = final_sampled_values.at[distribution_index, :].set(
            intervals[distribution_index, uniform_value_to_interval]
        )
    assert final_sampled_values.shape == (
        intervals.shape[0],
        sample_count_per_distribution,
    )
    return final_sampled_values
