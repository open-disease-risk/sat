# Development Guide

This document provides guidelines and best practices for developers contributing to the SAT project.

## Code Quality Standards

The SAT project maintains high standards of code quality through automated tools and processes.

### Pre-commit Hooks

We use [pre-commit](https://pre-commit.com/) to ensure code quality checks run automatically before each commit. This helps catch issues early and maintains consistent code style across contributors.

#### Installation

To install pre-commit hooks:

```bash
# Install pre-commit
pip install pre-commit

# Install the hooks
pre-commit install
```

#### Available Hooks

Our pre-commit configuration includes the following hooks:

1. **Basic checks** (trailing whitespace, file endings, etc.)
2. **Black** (code formatting)
3. **isort** (import sorting)
4. **Ruff** (fast linter, replacement for flake8)
5. **mypy** (static type checking)
6. **Bandit** (security checks)
7. **nbstripout** (cleans Jupyter notebook outputs)

#### Running Manually

To run all pre-commit hooks manually on all files:

```bash
pre-commit run --all-files
```

To run a specific hook:

```bash
pre-commit run black --all-files
```

### Type Checking with mypy

We use [mypy](https://mypy.readthedocs.io/) for static type checking. The configuration is in `pyproject.toml`.

To run mypy:

```bash
mypy sat
```

### Linting with Ruff

[Ruff](https://github.com/charliermarsh/ruff) is a fast Python linter. We use it as a replacement for flake8.

To run Ruff:

```bash
ruff check .
```

To apply fixes automatically:

```bash
ruff check --fix .
```

## Development Workflow

### Setting Up Development Environment

1. Clone the repository
2. Install poetry: `pip install poetry`
3. Install dependencies: `poetry install`
4. Install pre-commit hooks: `pre-commit install`

### Pull Request Guidelines

1. Create a feature branch from `main`
2. Implement your changes, ensuring tests pass
3. Ensure pre-commit hooks run without errors
4. Create a pull request with a clear description

### Testing

Run the test suite to ensure your changes don't break existing functionality:

```bash
# Run the full test suite
pytest

# Run with coverage report
pytest --cov=sat
```

## CI/CD Pipeline

Our CI/CD pipeline runs on every pull request and includes:

1. Running all tests
2. Running all linters and type checks
3. Building and testing the package

## Documentation

When adding new features, please update the relevant documentation:

- Add docstrings to all public functions, classes, and methods
- Update any relevant markdown files in the `docs/` directory
- For major features, consider adding an example notebook

## Troubleshooting

### Common Issues with pre-commit

**Issue**: Black is failing with "would reformat" errors
**Solution**: This is expected; pre-commit is preventing you from committing unformatted code. Run `pre-commit run black --all-files` to format the code.

**Issue**: mypy is failing with type errors
**Solution**: Address the type issues, or if necessary, add appropriate type ignore comments. Use these sparingly.

**Issue**: Hooks are slow
**Solution**: pre-commit caches results, so subsequent runs should be faster. For large codebases, consider using targeted commits.

## Code Review Guidelines

When reviewing code, pay attention to:

1. Adherence to project style conventions
2. Proper test coverage for new features
3. Documentation quality
4. Performance implications
5. Security considerations
