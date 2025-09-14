# Validation Workflow

After large changes, before committing, before PRs, if I ask you to validate, or if there's a good chance that some of this is broken:

```bash
# 1. Type checking (standard mode)
pyright

# 2. Linting and formatting
ruff check
ruff format
ruff check --select I --fix  # Import sorting

# 3. Tests with warnings as errors
pytest -v --tb=short -W error::UserWarning
```

Make sure that all are fixed before continuing. Run them again after fixes to make sure that fixing one doesn't break the others.
