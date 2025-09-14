# Python Exception Handling & Logging

## Core Principle

ALWAYS let exceptions percolate up by default. Do NOT suppress errors.

Only suppress an exception and use `_log.warning(...)` if EVERYTHING the user asked for can still be done. If any part of it is compromised, raise instead. Warnings are for recoverable conditions.

## Exception Handling Patterns

```python
import json
import subprocess
from logging import getLogger
from pathlib import Path

_log = getLogger(__name__)

def read_config(config_path: Path) -> dict[str, str]:
    """Read configuration file.

    @param config_path: Path to configuration file
    @return: Configuration dictionary
    @raises FileNotFoundError: If config file doesn't exist
    @raises ValueError: If config format is invalid
    """
    # Conserve exception tracebacks with 'from'
    try:
        with config_path.open() as config_file:
            configuration_data = json.load(config_file)
    except json.JSONDecodeError as json_error:
        raise ValueError(f"Invalid JSON in config file {config_path}: {json_error}") from json_error
    except OSError as os_error:
        raise FileNotFoundError(f"Cannot read config file {config_path}") from os_error
    # NO `except Exception`, let exceptions you didn't plan for percolate up

    _log.debug(f"Loaded config from {config_path}")
    return configuration_data
```

## Input Validation

```python
# Input validation with helpful messages
def calculate_average(input_values: tuple[float, ...]) -> float:
    """Calculate average of numeric values."""
    if len(input_values) == 0:
        raise ValueError("Cannot calculate average of empty sequence")
    if any(current_value < 0 for current_value in input_values):
        negative_values = [current_value for current_value in input_values if current_value < 0]
        raise ValueError(f"All values must be non-negative, got: {negative_values}")
    return sum(input_values) / len(input_values)
```

## Main Function Pattern

```python
def main():
    # NO global `try: ... except Exception: ...` that just prints errors, let Python show a stack trace
    config = read_config("./config_path.json")
```