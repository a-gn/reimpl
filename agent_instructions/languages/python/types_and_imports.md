# Python Type System & Language Features

## Type Safety Requirements

Type-check everything thoroughly, including type parameters, recursively. Use immutable types by default.

## Module Structure

```python
"""Module documentation."""

# Imports ALWAYS at the top
# No optional imports, no `except ImportError`, just let Python handle it
# Nothing between import lines, all imports in one block, so that import sorters work
# Standard library, then external packages, then project-internal stuff
from collections.abc import Mapping, Sequence
from pathlib import Path

from external_lib import external_thing

from current_project import internal_thing
```

## Immutable Types and Defaults

```python
# Immutable argument types and defaults: `Sequence[]` or `tuple[]` over `list[]`, for example
# Only use `| None` if the `None` case has a meaning that the original type can't represent. The empty case is well-represented by an empty collection here
def process_items(input_items: tuple[str, ...] = ()) -> dict[str, int]:
    """Process items with modern typing and immutable default."""
    item_length_mapping: dict[str, int] = {}
    for current_item in input_items:
        item_length_mapping[current_item] = len(current_item)
    return item_length_mapping

# Here the absence of a collection is different from an empty collection, it has meaning. We use `| None`
def process_optional_items(input_items: tuple[str, ...] | None) -> dict[str, int]:
    """When None is meaningful, handle explicitly."""
    if input_items is None:
        return {}
    return {current_item: len(current_item) for current_item in input_items}
```

## Collection Type Guidelines

```python
# Type system examples - when to use Sequence vs specific types
def process_various_collections(
    integer_values: Sequence[int],        # Good: int is not a sequence
    float_measurements: Sequence[float],  # Good: float is not a sequence
    byte_data: Sequence[bytes],          # Good: bytes is not a sequence
    name_list: tuple[str, ...],          # Immutable - avoid Sequence[str] since str is itself a Sequence[str]
    config_mapping: Mapping[str, int]    # Good: use Mapping for dict-like inputs
) -> tuple[tuple[int, ...], dict[str, float]]:
    """Process various collection types showing proper type hints.

    Pure function with immutable inputs and outputs.

    @param integer_values: Any sequence of integers (list, tuple, etc.)
    @param float_measurements: Any sequence of floats
    @param byte_data: Any sequence of byte strings
    @param name_list: Immutable tuple of strings (avoid Sequence[str])
    @param config_mapping: Any mapping from strings to integers
    @return: Processed integers and float statistics (both immutable)
    """
    # Process sequences where element type is not itself a sequence
    processed_ints = tuple(value * 2 for value in integer_values)

    # For strings, be specific about tuple to avoid sequence confusion and ensure immutability
    cleaned_names = tuple(name.strip() for name in name_list if len(name.strip()) > 0)

    # Mapping allows dict, defaultdict, etc. - return new dict (pure function)
    statistics = {key: float(value) for key, value in config_mapping.items()}

    return processed_ints, statistics
```

## Mutable Function Parameters

```python
# Non-pure function: make it clear what the side-effects are. Mutable argument type is necessary, so accepted
def add_integer_sequence_to_list(list_to_modify: list[float], maximum_integer: int) -> None:
    """Modify a list in-place to add numbers to it.

    @param list_to_modify The collection to which we will add numbers. It will be modified in place.
    @param maximum_integer We will add integers from 0 to this one, included. Must be strictly positive.
    """

    if maximum_integer <= 0:
        raise ValueError(f"maximum_integer must be strictly positive, we got {maximum_integer}")
    list_to_modify.extend(range(0, maximum_integer + 1))
```