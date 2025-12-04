# Installation

## Requirements

- Python 3.10 or higher
- gymnasium >= 1.0.0
- numpy >= 1.21.0

## Using uv (Recommended)

[uv](https://github.com/astral-sh/uv) is a fast Python package installer and resolver.

```bash
uv pip install gymnasium-retinatask
```

## Using pip

```bash
pip install gymnasium-retinatask
```

## From Source

For development or to get the latest features:

```bash
git clone https://github.com/Teaspoon-AI/gymnasium-retinatask.git
cd gymnasium-retinatask
uv pip install -e .
```

Or with pip:

```bash
git clone https://github.com/Teaspoon-AI/gymnasium-retinatask.git
cd gymnasium-retinatask
pip install -e .
```

## Development Installation

If you want to contribute or run tests:

```bash
git clone https://github.com/Teaspoon-AI/gymnasium-retinatask.git
cd gymnasium-retinatask
uv sync --all-groups
```

This installs the package with development dependencies including:
- pytest for testing
- black for code formatting
- isort for import sorting
- pyright for type checking

## Verifying Installation

Test your installation:

```python
import gymnasium as gym
import gymnasium_retinatask

env = gym.make("RetinaTask-v0")
print("Installation successful!")
env.close()
```

## Running Tests

After development installation:

```bash
# Run all tests
uv run pytest

# Run with verbose output
uv run pytest -v

# Run specific test file
uv run pytest src/gymnasium_retinatask/tests/test_retina_env.py
```

## Building Documentation

To build this documentation locally:

```bash
cd docs
uv run sphinx-build -b html . _build/html
```

Or using make:

```bash
cd docs
make html
```
