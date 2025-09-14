# Python Documentation Template

## Function Documentation Pattern

```python
from logging import getLogger

_log = getLogger(__name__)

def process_matrix(
    matrix_data: tuple[tuple[float, ...], ...],
    processing_axis: int = 0,
    should_normalize: bool = True
) -> tuple[tuple[tuple[float, ...], ...], dict[str, float]]:
    """Process 2D matrix with optional normalization.

    @param matrix_data: 2D matrix as tuple of rows. Shape: (n_rows, n_cols).
                        All rows must have same length. Immutable for safety.
    @param processing_axis: Axis along which to process (0=rows, 1=columns)
    @param should_normalize: Whether to normalize to [0, 1] range
    @return: Tuple of (processed_matrix, statistics_dict) where statistics contains
             'mean', 'std', and 'range' statistics
    @raises ValueError: If matrix_data is empty, ragged, or processing_axis is invalid
    """
    # Input validation with helpful messages
    if len(matrix_data) == 0:
        raise ValueError("Input matrix_data cannot be empty")

    row_lengths = [len(matrix_row) for matrix_row in matrix_data]
    if len(set(row_lengths)) > 1:
        raise ValueError(f"All rows must have same length, got lengths: {row_lengths}")

    if processing_axis not in (0, 1):
        raise ValueError(f"processing_axis must be 0 or 1, got {processing_axis}")

    _log.debug(f"Processing {len(matrix_data)}x{len(matrix_data[0])} matrix along axis={processing_axis}")

    # Implementation...
    processed_matrix_data: tuple[tuple[float, ...], ...] = ()
    processing_statistics: dict[str, float] = {"mean": 0.0, "std": 0.0, "range": 0.0}

    return processed_matrix_data, processing_statistics
```

## Attribution Requirements

Add "Originally written by [model_name] on YYYY/MM/DD" to:
- **New modules created by you**: At the end of the module's docstring
- **Large functions/classes**: In docstring (unless the outer scope already has it)

```python
"""Data processing utilities module.

Originally written by Claude Sonnet 4 on 2025/09/13
"""

class DataProcessor:
    """Main data processing class."""
    # No attribution here, already exists in the parent module

    def simple_helper(self, input_value: int) -> int:
        """Simple helper - no attribution needed."""
        return input_value * 2
```