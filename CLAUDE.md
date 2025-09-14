# Project-specific AI agent instructions

## External submodules

- Do not read or modify the `external` folder at the root of the project unless I ask you to.
- Do not run any checking/linting/formatting tools on `external` unless I ask you to do so.
- Always exclude `external` from commands that could modify it.

## Core Principles

• **Error Handling**: Never suppress errors or bypass failures; crash early or deliver everything requested
• **Dependencies**: Use standard library → project code → third-party (minimize additions)
• **Type Safety**: Type-check everything thoroughly, including type parameters, recursively
• **Input Validation**: Document constraints, validate inputs, provide helpful error messages
• **Simplicity**: Prefer short diffs, reuse existing code, remove lines by default, use pure functions, factor
• **Attribution**: Add "Originally written by [model] on YYYY/MM/DD" for large, new code units
• **Tool Installation**: Install cautiously, ask permission, adapt to environment

## Language-Specific Guidelines

@agent_instructions/languages/python/README.md
