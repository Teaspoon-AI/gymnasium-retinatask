# Contributing

Thank you for your interest in contributing to gymnasium-retinatask!

## Development Setup

1. Fork and clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/gymnasium-retinatask.git
cd gymnasium-retinatask
```

2. Install with development dependencies:

```bash
uv sync --all-groups
```

3. Verify installation:

```bash
uv run pytest
```

## Code Standards

### Formatting

We use `black` and `isort` for code formatting:

```bash
# Format code
uv run black src/

# Sort imports
uv run isort src/
```

### Type Checking

We use `pyright` for type checking:

```bash
uv run pyright src/
```

### Testing

All new features should include tests:

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=gymnasium_retinatask

# Run specific test
uv run pytest src/gymnasium_retinatask/tests/test_retina_env.py::TestRetinaEnv::test_reset
```

## Pull Request Process

1. Create a new branch for your feature:

```bash
git checkout -b feature/your-feature-name
```

2. Make your changes and add tests

3. Ensure all tests pass and code is formatted:

```bash
uv run pytest
uv run black src/
uv run isort src/
```

4. Commit with a clear message:

```bash
git commit -m "Add feature: description of your changes"
```

5. Push to your fork:

```bash
git push origin feature/your-feature-name
```

6. Open a Pull Request on GitHub

## Areas for Contribution

### Environment Enhancements

- Additional reward functions
- Alternative pattern sets
- Curriculum learning support
- Multi-objective variants

### Documentation

- Tutorial examples
- Integration guides (PyTorch, JAX, etc.)
- Benchmarking results
- Visualization tools

### Testing

- Additional test cases
- Performance benchmarks
- Integration tests
- Compatibility tests

### Examples

- Neural network examples (CNN, MLP, etc.)
- Evolutionary algorithm examples
- Reinforcement learning examples
- Visualization examples

## Documentation

To build documentation locally:

```bash
cd docs
uv run sphinx-build -b html . _build/html
```

View at `docs/_build/html/index.html`

## Questions?

- Open an issue for bugs or feature requests
- Discussions for questions and ideas
- Email maintainer for other inquiries

## Code of Conduct

Please be respectful and constructive in all interactions. We aim to maintain a welcoming community for all contributors.
