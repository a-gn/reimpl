"""Utilities for random sampling."""

from functools import partial

import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.typing as jt


@partial(jax.jit, static_argnames=["sample_count_per_distribution"])
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

    interval_lengths = intervals[:, 1:] - intervals[:, :-1]
    # compute the probability of each uniform interval in the entire distributions
    interval_probabilities = interval_lengths * pdf_values

    # sample uniform values within [0, 1], which we will use as positions within the input bounds weighted by PDF values
    position_sampling_key, key = jax.random.split(key)
    uniformly_sampled_positions_in_0_1 = jax.random.uniform(
        key,
        shape=(intervals.shape[0], sample_count_per_distribution),
        dtype=float,
        minval=0.0,
        maxval=1.0,
    )
    del position_sampling_key

    # find where intervals start in [0, 1]
    cumulative_interval_probabilities = jnp.cumulative_sum(
        interval_probabilities, axis=1, include_initial=True
    )

    def _auxiliary_per_distribution_computation(
        distribution_index: int, previous_results: jnp.ndarray
    ):
        # find where [0,1] positions fall within the cumulative sum of interval probabilities
        assigned_interval_lower_bound_indices = (
            jnp.searchsorted(
                cumulative_interval_probabilities[distribution_index, :],
                uniformly_sampled_positions_in_0_1[distribution_index, :],
                side="right",  # so we're always one above the interval's lower bound's position
            )
            - 1
        )
        assigned_interval_lower_bound_indices = jnp.where(
            assigned_interval_lower_bound_indices == -1,
            0,
            assigned_interval_lower_bound_indices,
        )

        assigned_interval_lower_bounds_0_1 = cumulative_interval_probabilities[
            distribution_index, assigned_interval_lower_bound_indices
        ]
        assigned_interval_lower_bounds_0_1 = jnp.where(
            assigned_interval_lower_bounds_0_1 == -1,
            0,
            assigned_interval_lower_bounds_0_1,
        )

        # convert to final unit: shift to [0, x), scale down by interval probability then up by interval length, then
        # shift to interval lower bound
        final_values_for_this_distribution = (
            (
                uniformly_sampled_positions_in_0_1[distribution_index]
                - assigned_interval_lower_bounds_0_1
            )
            / interval_probabilities[
                distribution_index, assigned_interval_lower_bound_indices
            ]
            * interval_lengths[
                distribution_index, assigned_interval_lower_bound_indices
            ]
        ) + intervals[distribution_index, assigned_interval_lower_bound_indices]

        return previous_results.at[distribution_index].set(
            final_values_for_this_distribution
        )

    # for each distribution, assign [0, 1] values to their interval, then convert them to the final unit
    final_sampled_values = lax.fori_loop(
        0,
        pdf_values.shape[0],
        _auxiliary_per_distribution_computation,
        jnp.zeros_like(
            intervals, shape=(intervals.shape[0], sample_count_per_distribution)
        ),
    )

    return final_sampled_values
