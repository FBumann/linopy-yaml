# CLAUDE.md

## Project Overview

`linopy_yaml` is a YAML-based math definition layer for [linopy](https://github.com/PyPSA/linopy).
It lets users define optimisation problems declaratively in YAML and build them into `linopy.Model` objects at runtime.

See `SPEC.md` for the full design specification.

## Common Commands

```bash
# Install in dev mode
pip install -e .[dev]

# Run tests
pytest

# Lint and format
ruff check .
ruff check --fix .
ruff format .

# Type check
mypy linopy_yaml
```

## Package Structure

```
linopy_yaml/
├── __init__.py          # Public API: Model, register
├── schema.py            # Pydantic models for YAML validation
├── loader.py            # Data coercion, validation, master coords
├── expression_parser.py # pyparsing grammar for math expressions
├── where_parser.py      # pyparsing grammar for where strings
├── builder.py           # Schema + data → linopy Model construction
├── helpers.py           # Built-in helpers (sum, roll) + registry
└── model.py             # Model subclass with from_yaml(), extend()
tests/
├── conftest.py          # Shared fixtures
├── test_schema.py       # YAML schema validation tests
├── test_loader.py       # Data loading and coercion tests
├── test_parser.py       # Expression and where-string parser tests
├── test_builder.py      # End-to-end model building tests
└── test_dispatch.py     # Integration test with the dispatch example
```

## Development Guidelines

- This package is a **pure consumer** of linopy's public API. Never depend on linopy internals.
- All validation should happen at load time with clear, actionable error messages.
- Use `ruff` for linting/formatting, `mypy` for type checking, `pytest` for tests.
- Keep the dependency footprint minimal.
