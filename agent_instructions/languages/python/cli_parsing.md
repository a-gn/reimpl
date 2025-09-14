# Python CLI with Click

## Click Pattern

```python
import click
from pathlib import Path
from logging import getLogger

_log = getLogger(__name__)

@click.command()
@click.option('--input-file', type=click.Path(exists=True, path_type=Path), required=True)
@click.option('--threshold', type=float, default=0.5, help='Threshold (0.0-1.0)')
@click.option('--output-dir', type=click.Path(path_type=Path), help='Output directory')
def main(input_file: Path, threshold: float, output_dir: Path | None) -> None:
    """Process input file with threshold filtering.

    Click options must be: required=True, have default, or be | None.
    """
    # Explicit type conversion and validation (reassign since types match)
    input_file = Path(input_file)  # Ensure proper Path type
    threshold = float(threshold)   # Explicit conversion
    output_dir = Path(output_dir) if output_dir is not None else None

    if not 0.0 <= threshold <= 1.0:
        raise click.BadParameter(f'Threshold must be 0.0-1.0, got {threshold}')

    _log.debug(f"Processing {input_file} with threshold={threshold}")

    # Use converted types for type safety
    process_file(input_file, threshold, output_dir)
```