# Python Project Setup with uv

For new Python projects, use `uv` for fast dependency management:

```bash
# Create new project
uv init my-project
cd my-project

# Add dependencies if the user requested some
uv add click pathlib-extensions

# Add development dependencies (check `validation.md` and `testing.md` for the up-to-date list)
uv add --dev pytest pyright ruff

# Install and run
uv run python -m my_project
```
