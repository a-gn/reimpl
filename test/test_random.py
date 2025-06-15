"""Unit tests for piecewise uniform sampling.

Written by Claude Sonnet 4.

"""

import jax
import jax.numpy as jnp
import pytest

from reimpl_a_gn.threed.random import piecewise_uniform


class TestPiecewiseUniform:
    """Test cases for piecewise_uniform function."""

    def test_single_distribution_single_interval(self) -> None:
        """Test sampling from a single distribution over two intervals."""
        key = jax.random.PRNGKey(42)
        intervals = jnp.array([[0.0, 0.5, 1.0]])
        pdf_values = jnp.array([[1.3, 0.7]])
        sample_count = 100

        samples = piecewise_uniform(key, intervals, pdf_values, sample_count)

        assert samples.shape == (1, sample_count)
        # All samples should be in [0, 1]
        assert jnp.all(samples >= 0.0)
        assert jnp.all(samples <= 1.0)
        # Should have some spread (not all the same value)
        assert jnp.std(samples) > 0.01

    def test_single_distribution_multiple_intervals_uniform(self) -> None:
        """Test sampling from multiple intervals with equal probability density."""
        key = jax.random.PRNGKey(123)
        intervals = jnp.array([[0.0, 1.0, 2.0]])  # Two intervals: [0,1], [1,2]
        pdf_values = jnp.array([[0.5, 0.5]])  # Equal density in both intervals
        sample_count = 1000

        samples = piecewise_uniform(key, intervals, pdf_values, sample_count)

        assert samples.shape == (1, sample_count)
        assert jnp.all(samples >= 0.0)
        assert jnp.all(samples <= 2.0)

        # Should have roughly equal samples in each interval
        samples_in_first_interval = jnp.sum((samples >= 0.0) & (samples < 1.0))
        samples_in_second_interval = jnp.sum((samples >= 1.0) & (samples <= 2.0))

        # With equal probabilities, expect roughly 50/50 split (within statistical tolerance)
        assert (
            abs(samples_in_first_interval - samples_in_second_interval)
            < sample_count * 0.1
        )

    def test_single_distribution_multiple_intervals_nonuniform(self) -> None:
        """Test sampling from multiple intervals with different probability densities."""
        key = jax.random.PRNGKey(456)
        intervals = jnp.array([[0.0, 1.0, 2.0]])  # Two unit-length intervals
        pdf_values = jnp.array([[0.8, 0.2]])  # 80% probability in first, 20% in second
        sample_count = 1000

        samples = piecewise_uniform(key, intervals, pdf_values, sample_count)

        assert samples.shape == (1, sample_count)

        samples_in_first_interval = jnp.sum((samples >= 0.0) & (samples < 1.0))
        samples_in_second_interval = jnp.sum((samples >= 1.0) & (samples <= 2.0))

        # Should have roughly 4:1 ratio (80:20)
        expected_first = sample_count * 0.8
        expected_second = sample_count * 0.2

        assert abs(samples_in_first_interval - expected_first) < sample_count * 0.1
        assert abs(samples_in_second_interval - expected_second) < sample_count * 0.1

    def test_multiple_distributions(self) -> None:
        """Test sampling from multiple different distributions simultaneously."""
        key = jax.random.PRNGKey(789)
        intervals = jnp.array(
            [
                [0.0, 1.0, 2.0],  # First distribution: two intervals
                [
                    0.0,
                    0.5,
                    1.0,
                ],  # Second distribution: two intervals of different lengths
            ]
        )
        pdf_values = jnp.array(
            [
                [0.5, 0.5],  # First: equal density
                [
                    1.0,
                    1.0,
                ],  # Second: equal density (but intervals have different lengths)
            ]
        )
        sample_count = 500

        samples = piecewise_uniform(key, intervals, pdf_values, sample_count)

        assert samples.shape == (2, sample_count)

        # First distribution samples should be in [0, 2]
        assert jnp.all(samples[0] >= 0.0)
        assert jnp.all(samples[0] <= 2.0)

        # Second distribution samples should be in [0, 1]
        assert jnp.all(samples[1] >= 0.0)
        assert jnp.all(samples[1] <= 1.0)

    def test_different_interval_lengths_equal_probability(self) -> None:
        """Test that intervals of different lengths with equal total probability work correctly."""
        key = jax.random.PRNGKey(321)
        intervals = jnp.array(
            [[0.0, 1.0, 3.0]]
        )  # Intervals: [0,1] length 1, [1,3] length 2
        # For equal probability: first interval needs density 0.5, second needs density 0.25
        pdf_values = jnp.array([[0.5, 0.25]])  # 0.5*1 = 0.5, 0.25*2 = 0.5
        sample_count = 1000

        samples = piecewise_uniform(key, intervals, pdf_values, sample_count)

        samples_in_first = jnp.sum((samples >= 0.0) & (samples < 1.0))
        samples_in_second = jnp.sum((samples >= 1.0) & (samples <= 3.0))

        # Should be roughly equal despite different interval lengths
        assert abs(samples_in_first - samples_in_second) < sample_count * 0.1

    def test_zero_probability_interval(self) -> None:
        """Test handling of intervals with zero probability."""
        key = jax.random.PRNGKey(654)
        intervals = jnp.array([[0.0, 1.0, 2.0]])
        pdf_values = jnp.array([[1.0, 0.0]])  # All probability in first interval
        sample_count = 100

        samples = piecewise_uniform(key, intervals, pdf_values, sample_count)

        # All samples should be in first interval only
        assert jnp.all(samples >= 0.0)
        assert jnp.all(samples < 1.0)

    def test_invalid_intervals_shape(self) -> None:
        """Test error handling for invalid intervals shape."""
        key = jax.random.PRNGKey(42)

        # 1D intervals (should be 2D)
        with pytest.raises(
            ValueError,
            match="intervals should have shape \\(N, interval_count \\+ 1\\)",
        ):
            piecewise_uniform(key, jnp.array([0.0, 1.0]), jnp.array([[1.0]]), 10)

        # Too few interval bounds
        with pytest.raises(
            ValueError, match="intervals should have a second axis of size at least 2"
        ):
            intervals = jnp.array([[0.0]])  # Only one bound, can't define an interval
            pdf_values = jnp.array([[]])  # Empty pdf values
            piecewise_uniform(key, intervals, pdf_values, 10)

    def test_mismatched_pdf_values_shape(self) -> None:
        """Test error handling for mismatched pdf_values shape."""
        key = jax.random.PRNGKey(42)
        intervals = jnp.array([[0.0, 1.0, 2.0]])  # 2 intervals

        # Wrong number of pdf values
        pdf_values = jnp.array([[1.0]])  # Only 1 value for 2 intervals
        with pytest.raises(ValueError, match="pdf_values should have shape"):
            piecewise_uniform(key, intervals, pdf_values, 10)

        # Wrong dimensionality
        pdf_values = jnp.array([1.0, 0.0])  # 1D instead of 2D
        with pytest.raises(ValueError, match="pdf_values should have shape"):
            piecewise_uniform(key, intervals, pdf_values, 10)

    def test_negative_pdf_values(self) -> None:
        """Test error handling for negative probability density values."""
        key = jax.random.PRNGKey(42)
        intervals = jnp.array([[0.0, 1.0, 2.0]])
        pdf_values = jnp.array([[0.5, -0.5]])  # Negative density

        with pytest.raises(
            ValueError, match="probability density values can't be strictly negative"
        ):
            piecewise_uniform(key, intervals, pdf_values, 10)

    def test_pdf_values_not_summing_to_one(self) -> None:
        """Test error handling when probability densities don't sum to 1."""
        key = jax.random.PRNGKey(42)
        intervals = jnp.array([[0.0, 1.0, 2.0]])

        # Sum to 0.5 instead of 1.0
        pdf_values = jnp.array([[0.25, 0.25]])
        with pytest.raises(
            ValueError, match="the sum of interval probabilities should equal one"
        ):
            piecewise_uniform(key, intervals, pdf_values, 10)

        # Sum to 2.0 instead of 1.0
        pdf_values = jnp.array([[1.0, 1.0]])
        with pytest.raises(
            ValueError, match="the sum of interval probabilities should equal one"
        ):
            piecewise_uniform(key, intervals, pdf_values, 10)

    def test_reproducibility_with_same_key(self) -> None:
        """Test that same key produces same results."""
        key = jax.random.PRNGKey(42)
        intervals = jnp.array([[0.0, 1.0]])
        pdf_values = jnp.array([[1.0]])
        sample_count = 10

        samples1 = piecewise_uniform(key, intervals, pdf_values, sample_count)
        samples2 = piecewise_uniform(key, intervals, pdf_values, sample_count)

        assert jnp.allclose(samples1, samples2)

    def test_different_results_with_different_keys(self) -> None:
        """Test that different keys produce different results."""
        intervals = jnp.array([[0.0, 0.5, 1.0]])
        pdf_values = jnp.array([[1.3, 0.7]])
        sample_count = 100

        key1 = jax.random.PRNGKey(42)
        key2 = jax.random.PRNGKey(43)

        samples1 = piecewise_uniform(key1, intervals, pdf_values, sample_count)
        samples2 = piecewise_uniform(key2, intervals, pdf_values, sample_count)

        # Should be different (extremely unlikely to be identical by chance)
        assert not jnp.allclose(samples1, samples2)

    def test_edge_case_very_small_intervals(self) -> None:
        """Test handling of very small intervals."""
        key = jax.random.PRNGKey(42)
        intervals = jnp.array([[0.0, 1e-10, 1.0]])  # Very small first interval
        # Compensate with high density in small interval for equal probability
        pdf_values = jnp.array([[5e9, 0.5]])  # 5e9 * 1e-10 = 0.5, 0.5 * 1 = 0.5
        sample_count = 1000

        samples = piecewise_uniform(key, intervals, pdf_values, sample_count)

        assert samples.shape == (1, sample_count)
        # Should still work numerically despite small interval
        assert jnp.all(samples >= 0.0)
        assert jnp.all(samples <= 1.0)

    def test_samples_are_actually_within_sampled_intervals(self) -> None:
        """Test that returned samples are actually within the intervals they were sampled from.

        This test specifically checks that the function doesn't just return interval bounds
        but actual samples from within the intervals.
        """
        key = jax.random.PRNGKey(42)
        intervals = jnp.array([[0.0, 1.0, 2.0]])
        pdf_values = jnp.array([[0.5, 0.5]])
        sample_count = 100

        samples = piecewise_uniform(key, intervals, pdf_values, sample_count)

        # Check that we don't just get the interval bounds
        unique_samples = jnp.unique(samples)

        # With 100 samples from continuous distributions, we should have many unique values
        # (not just the 3 interval boundary values)
        assert len(unique_samples) > 10  # Much more than just boundary values

        # Check that some samples are strictly inside intervals (not on boundaries)
        samples_strictly_inside_first = jnp.sum((samples > 0.0) & (samples < 1.0))
        samples_strictly_inside_second = jnp.sum((samples > 1.0) & (samples < 2.0))

        # Should have samples strictly inside intervals, not just on boundaries
        assert samples_strictly_inside_first > 0
        assert samples_strictly_inside_second > 0
