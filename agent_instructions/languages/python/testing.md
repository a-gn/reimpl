# Python Testing Guidelines

## Core Testing Principles

- Use pytest
- Use fixtures and parametrization when possible
- Keep tests near-instant
- **DO NOT MOCK WHEN YOU CAN DO WITHOUT IT**
- Prefer injection (update APIs to inject dependencies with default values for non-testing cases)
- Type fixture results in function parameters
- Use data that's as small as possible while testing what we want
- Test both main cases and interesting/risky/edge/error cases
- Prefer dumb data like `np.zeros` or `np.ones`
- Use NumPy RNG with fixed seed for deterministic non-constant data
- Keep things simple
- Compare entire structures strictly when possible

## Injection over Mocking

```python
from collections.abc import Sequence

import numpy as np

# Good: Dependency injection with meaningful defaults (avoid None defaults)
def process_data(
    input_data: Sequence[float],
    random_generator: np.random.Generator = np.random.default_rng()
) -> tuple[float, ...]:
    """Process data with injected RNG (unseeded for production randomness).

    @param input_data: Sequence of float values to process
    @param random_generator: Random number generator for adding noise
    @return: Processed values with added noise as immutable tuple
    """
    noise = random_generator.normal(0, 0.1, len(input_data))
    return tuple(value + noise_val for value, noise_val in zip(input_data, noise))

# Test with injected dependency
def test_process_data_deterministic():
    test_rng = np.random.Generator(np.random.PCG64(123))  # Fixed algorithm and seed for testing
    input_data = (1.0, 2.0, 3.0)

    result = process_data(input_data, random_generator=test_rng)

    # Generate expected values with same RNG configuration
    expected_rng = np.random.Generator(np.random.PCG64(123))
    expected_noise = expected_rng.normal(0, 0.1, len(input_data))
    expected = tuple(value + noise_val for value, noise_val in zip(input_data, expected_noise))

    assert np.allclose(result, expected, atol=1e-10)
```

## Fixtures and Parametrization

```python
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest

@pytest.fixture
def sample_matrix() -> np.ndarray:
    """Provide deterministic test matrix."""
    return np.array([[1.0, 2.0], [3.0, 4.0]])

@pytest.fixture
def temp_config_file():
    """Create temporary config file for testing."""
    with TemporaryDirectory() as temp_dir:
        config_path = Path(temp_dir) / "config.json"
        config_data = {"threshold": 0.5, "max_iterations": 100}
        config_path.write_text(json.dumps(config_data))
        yield config_path

@pytest.mark.parametrize(["input_value", "expected"], [
    (0.0, 0.0),
    (1.0, 1.0),
    (-1.0, 1.0),
    (2.0, 4.0),
])
def test_square_function(input_value: float, expected: float):
    """Verify that square function correctly computes x²."""
    result = square(input_value)
    assert result == expected

def test_matrix_processing(sample_matrix: np.ndarray):
    """Verify that matrix processing doubles all values."""
    result = process_matrix(sample_matrix)
    expected = np.array([[2.0, 4.0], [6.0, 8.0]])

    # Compare entire structure strictly
    assert np.allclose(result, expected)
```

## Deterministic Data Generation

```python
import numpy as np
import torch

def test_with_random_data():
    """Generate deterministic random data for testing."""
    # Always use NumPy RNG with fixed algorithm and seed
    rng = np.random.Generator(np.random.PCG64(42))

    # Generate test data with NumPy
    test_matrix = rng.normal(0, 1, (10, 5))
    test_labels = rng.integers(0, 3, 10)

    # Convert to target library (example with PyTorch)
    test_matrix_torch = torch.from_numpy(test_matrix)
    test_labels_torch = torch.from_numpy(test_labels)

    result = train_model(test_matrix_torch, test_labels_torch)

    # Test specific expected behavior with this deterministic data
    assert result.accuracy > 0.8
    assert len(result.weights) == 5

def test_simple_data_preferred():
    """Use simple data when possible."""
    # Prefer simple, obvious data
    zeros_input = np.zeros((3, 3))
    ones_input = np.ones((2, 4))

    zeros_result = normalize_matrix(zeros_input)
    ones_result = normalize_matrix(ones_input)

    # Simple assertions on simple data
    assert np.allclose(zeros_result, np.zeros((3, 3)))
    assert np.allclose(ones_result, np.ones((2, 4)))
```

## Error and Edge Case Testing

```python
def test_empty_input_raises_error():
    """Verify that processing empty input raises descriptive ValueError."""
    with pytest.raises(ValueError, match="Input cannot be empty"):
        process_empty_input([])

def test_invalid_type_raises_error():
    """Verify that non-numeric input raises descriptive TypeError."""
    with pytest.raises(TypeError, match="Expected numeric input"):
        process_invalid_type("not_a_number")

def test_very_small_numbers():
    """Verify that function handles extremely small floating-point values correctly."""
    tiny_result = process_data((1e-10, 1e-15))
    assert len(tiny_result) == 2
    assert all(isinstance(x, float) for x in tiny_result)

def test_boundary_conditions():
    """Test interesting boundary cases."""
    # Test exactly at boundaries
    assert is_valid_percentage(0.0) is True
    assert is_valid_percentage(1.0) is True
    assert is_valid_percentage(-0.0001) is False
    assert is_valid_percentage(1.0001) is False
```

## Strict Structure Comparison

```python
def test_json_structure_comparison():
    """Compare entire JSON structures strictly."""
    input_data = {"users": [{"id": 1, "name": "Alice"}], "count": 1}

    result = transform_user_data(input_data)
    expected = {
        "users": [{"id": 1, "name": "Alice", "status": "active"}],
        "count": 1,
        "processed": True
    }

    # Compare entire structure - don't assert individual fields
    assert result == expected

def test_array_comparison():
    """Compare arrays with appropriate tolerance."""
    input_array = np.array([1.0, 2.0, 3.0])

    result = apply_transformation(input_array)
    expected = np.array([2.0, 4.0, 6.0])

    # Single comparison instead of element-by-element assertions
    assert np.allclose(result, expected, atol=1e-6)

def test_complex_structure():
    """Test complex nested structures."""
    result = build_analysis_report(sample_data)
    expected = {
        "summary": {"total": 100, "average": 5.5},
        "details": [{"category": "A", "values": [1, 2, 3]}],
        "metadata": {"version": "1.0", "timestamp": "2025-01-01"}
    }

    # One assertion for the entire structure
    assert result == expected
```

## Async Testing

Use `pytest-asyncio` for testing async functions. Install it and add to dev dependencies if needed.

```python
import asyncio

import pytest

@pytest.mark.asyncio
async def test_async_function():
    """Verify that async data fetching returns expected structure."""
    result = await fetch_data_async("test_url")
    expected = {"status": "success", "data": [1, 2, 3]}

    assert result == expected

@pytest.mark.asyncio
async def test_async_with_fixture(sample_matrix: np.ndarray):
    """Verify that async matrix processing doubles all values."""
    processed = await process_matrix_async(sample_matrix)
    expected = np.array([[2.0, 4.0], [6.0, 8.0]])

    assert np.allclose(processed, expected)

# Only use asyncio.run() if you specifically need to control the event loop
def test_async_with_custom_loop():
    """Verify behavior when custom event loop control is required."""
    async def custom_async_test():
        # Custom setup that requires specific loop behavior
        result = await some_function_requiring_custom_loop()
        return result

    result = asyncio.run(custom_async_test())
    assert result is not None
```