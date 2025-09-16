"""Quick utility to export JAX arrays to CSV files.

Originally written by Claude on 2025/09/17
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np


def array_to_csv(array: jnp.ndarray, filename: str | Path) -> None:
    """Export a JAX array to CSV file for spreadsheet software.

    @param array: JAX array to export (1D or 2D)
    @param filename: Output CSV file path
    @raises ValueError: If array has more than 2 dimensions
    """
    if array.ndim > 2:
        raise ValueError(
            f"Cannot export {array.ndim}D array to CSV. Use 1D or 2D arrays only."
        )

    # Convert to numpy for compatibility with savetxt
    np_array = np.asarray(array)

    # Ensure 2D for CSV export
    if np_array.ndim == 1:
        np_array = np_array.reshape(-1, 1)

    np.savetxt(filename, np_array, delimiter=",", fmt="%.6g")
