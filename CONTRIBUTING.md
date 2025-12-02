# Contributing to Gymnasium Retina Task

Thank you for your interest in contributing to Gymnasium Retina Task! This document provides guidelines for contributing to the project.

## Code of Conduct

This project adheres to the Farama Foundation Code of Conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to the Farama Foundation moderators.

## Development Setup

### Prerequisites

- Python 3.10 or higher
- `uv` package manager (recommended) or `pip`

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Teaspoon-AI/gymnasium-retinatask.git
   cd gymnasium-retinatask
   ```

2. Install in development mode with all dependencies:
   ```bash
   uv sync --all-groups
   ```

   Or with pip:
   ```bash
   pip install -e ".[dev,docs]"
   ```

3. Install pre-commit hooks:
   ```bash
   uv run pre-commit install
   ```

## Development Workflow

### Before Making Changes

1. Create a new branch for your feature or bugfix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make sure all tests pass:
   ```bash
   uv run pytest
   ```

### Making Changes

1. Write your code following the project's style guidelines (see below)

2. Add or update tests for your changes

3. Update documentation if needed

4. Run the pre-commit checks:
   ```bash
   uv run pre-commit run --all-files
   ```

5. Run the test suite:
   ```bash
   uv run pytest
   ```

### Code Style

This project uses the following tools for code quality:

- **Black** for code formatting
- **isort** for import sorting
- **flake8** for linting
- **pyright** for type checking
- **pydocstyle** for docstring style (Google convention)

All of these are automatically run via pre-commit hooks.

### Docstring Style

We use Google-style docstrings. Example:

```python
def my_function(arg1: int, arg2: str) -> bool:
    """Brief description of the function.

    Longer description if needed.

    Args:
        arg1: Description of arg1.
        arg2: Description of arg2.

    Returns:
        Description of return value.

    Raises:
        ValueError: Description of when this is raised.
    """
    pass
```

### Type Hints

All public functions should have type hints for parameters and return values.

## Testing

### Running Tests

Run all tests:
```bash
uv run pytest
```

Run specific test file:
```bash
uv run pytest src/gymnasium_retinatask/tests/test_retina_env.py
```

Run with coverage:
```bash
uv run pytest --cov=gymnasium_retinatask --cov-report=html
```

### Writing Tests

- Place test files in `src/gymnasium_retinatask/tests/`
- Use descriptive test names: `test_<what_is_being_tested>`
- Group related tests in classes
- Use pytest fixtures for common setup

## Documentation

### Building Documentation

```bash
cd docs
uv run sphinx-build -b html . _build/html
```

View the built docs:
```bash
open _build/html/index.html
```

### Auto-rebuild on changes

```bash
cd docs
uv run sphinx-autobuild . _build/html
```

## Pull Request Process

1. Update the CHANGELOG.md with details of your changes

2. Ensure all tests pass and pre-commit checks are clean

3. Update documentation if you've changed the API

4. Create a pull request with a clear title and description

5. Link any related issues in the PR description

6. Wait for review and address any feedback

### PR Checklist

- [ ] Tests pass locally
- [ ] Pre-commit hooks pass
- [ ] Documentation is updated
- [ ] CHANGELOG.md is updated
- [ ] Code follows project style guidelines
- [ ] Commit messages are clear and descriptive

## Reporting Bugs

When reporting bugs, please include:

1. A clear, descriptive title
2. Steps to reproduce the issue
3. Expected behavior
4. Actual behavior
5. Python version and operating system
6. Relevant code snippets or error messages

## Suggesting Enhancements

Enhancement suggestions are welcome! Please:

1. Use a clear, descriptive title
2. Provide a detailed description of the proposed feature
3. Explain why this enhancement would be useful
4. Provide examples of how it would be used

## Questions?

If you have questions about contributing, feel free to:

- Open an issue for discussion
- Reach out to the maintainers

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
