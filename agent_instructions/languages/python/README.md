### Python

#### Core Requirements

##### Error Handling
ALWAYS let exceptions percolate up by default. Do NOT suppress errors.

Only suppress an exception and use `_log.warning(...)` if EVERYTHING the user asked for can still be done. If any part is compromised, raise instead. Warnings are for recoverable conditions.

- Conserve exception tracebacks with `from`
- Provide helpful error messages with context
- NO global `try: ... except Exception: ...` that just prints errors
- Let Python show stack traces for unexpected errors

##### Documentation
Function documentation pattern:
```python
def process_data(input_data: tuple[str, ...], normalize: bool = True) -> dict[str, float]:
    """Process input data with optional normalization.

    @param input_data: Input strings to process. Immutable for safety.
    @param normalize: Whether to normalize results to [0, 1] range
    @return: Mapping of processed results
    @raises ValueError: If input_data is empty or contains invalid values
    """
```

Add "Originally written by [model] on YYYY/MM/DD" for new modules and large functions.

##### Type Safety
Type-check everything thoroughly, including type parameters, recursively. Use immutable types by default.

- Use `Sequence[T]` for sequences, `Mapping[K, V]` for dict-like inputs, unless another rule overrides this or you need mutability
- Avoid `Sequence[str]` since `str` is itself a `Sequence[str]` - use `tuple[str, ...]`, same rule for other types `T` such that `T` is also a `Sequence[T]`, like bytes
- Only use `T | None` when None has a special meaning different from `T`. For example, if an empty collection can say what you want, use that instead
- Imports ALWAYS at the top in three blocks: stdlib, external packages, then project-internal

#### Optional Specialized Files

Load these only when relevant to your current task:

- `@types_and_imports.md` - Detailed type system guidelines with comprehensive examples
- `@exceptions.md` - Exception handling patterns with validation examples
- `@documentation.md` - Complete documentation templates with attribution examples
- `@testing.md` - Pytest guidelines with fixtures, parametrization, and dependency injection (load when writing tests)
- `@subprocess.md` - Safe subprocess execution patterns (load when spawning processes)
- `@cli_parsing.md` - Click-based CLI patterns with type safety (load for CLI tools)
- `@validation.md` - Project validation workflow (pyright, ruff, pytest commands) for pull requests, large changes, or when the user explicitly requests it
- `@new_project_setup.md` - Project initialization and structure (load for new projects)
