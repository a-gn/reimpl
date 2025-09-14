# Python Subprocess Handling

## Safe Command Execution

```python
import json
import subprocess
from pathlib import Path
from logging import getLogger

_log = getLogger(__name__)

def run_command(command_arguments: tuple[str, ...], working_directory: Path | None = None) -> str:
    """Run shell command safely.

    @param command_arguments: Command and arguments to execute
    @param working_directory: Directory to run command in (None for current directory)
    @return: Command stdout output
    @raises subprocess.CalledProcessError: If command fails
    """
    # use JSON to log complex data precisely for reproduction
    _log.debug(f"Running command (logged as JSON): {json.dumps(command_arguments)}")

    # Always use check=True and capture output
    command_result = subprocess.run(
        command_arguments,
        cwd=working_directory,
        capture_output=True,
        text=True,
        check=True
    )

    return command_result.stdout.strip()
```