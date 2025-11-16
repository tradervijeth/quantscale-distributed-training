# Contributing to QuantScale

Thank you for your interest in contributing to QuantScale! This document provides guidelines and instructions for contributing.

## 👨‍💻 Author

**Vithushan Jeyapahan**
Co-Founder & ML Engineer, Vijeth Ltd
📧 finance@vijeth.com

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Reporting Bugs](#reporting-bugs)
- [Suggesting Enhancements](#suggesting-enhancements)

## Code of Conduct

This project adheres to a code of professional conduct. By participating, you are expected to:

- Be respectful and inclusive
- Focus on constructive feedback
- Accept criticism gracefully
- Prioritize the community's best interests

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/quantscale-distributed-training.git
   cd quantscale-distributed-training
   ```
3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/tradervijeth/quantscale-distributed-training.git
   ```

## Development Setup

### Prerequisites

- Python 3.10+
- CUDA 11.8+ (for GPU support)
- Git

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .

# Install development dependencies
pip install -e ".[dev]"
```

### Development Dependencies

```bash
pip install pytest pytest-cov black flake8 mypy isort pre-commit
```

### Pre-commit Hooks

We use pre-commit hooks to ensure code quality:

```bash
# Install pre-commit hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

## Coding Standards

### Python Style Guide

- Follow [PEP 8](https://pep8.org/)
- Maximum line length: 100 characters
- Use type hints for all functions
- Write Google-style docstrings

### Code Formatting

We use **Black** for code formatting:

```bash
black quantscale/ tests/ scripts/
```

### Import Sorting

We use **isort** for import sorting:

```bash
isort quantscale/ tests/ scripts/
```

### Linting

We use **flake8** for linting:

```bash
flake8 quantscale/ tests/ scripts/
```

### Type Checking

We use **mypy** for type checking:

```bash
mypy quantscale/ --ignore-missing-imports
```

### Docstring Format

Use Google-style docstrings:

```python
def function_name(param1: int, param2: str) -> bool:
    """
    Brief description of function.

    More detailed description if needed.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ValueError: Description of when this is raised

    Example:
        >>> result = function_name(42, "test")
        >>> print(result)
        True
    """
    pass
```

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=quantscale --cov-report=html

# Run specific test file
pytest tests/test_data.py -v

# Run in parallel
pytest tests/ -n auto
```

### Writing Tests

- Place tests in the `tests/` directory
- Name test files as `test_*.py`
- Name test functions as `test_*`
- Use pytest fixtures for setup/teardown
- Aim for >80% code coverage

Example:

```python
import pytest
from quantscale.models import TemporalFusionTransformer

def test_model_creation():
    """Test model instantiation."""
    model = TemporalFusionTransformer(
        input_dim=10,
        hidden_dim=64
    )
    assert model is not None
    assert model.input_dim == 10
```

## Pull Request Process

### Before Submitting

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the coding standards

3. **Write/update tests** for your changes

4. **Run tests** to ensure everything passes:
   ```bash
   pytest tests/ -v
   ```

5. **Format code**:
   ```bash
   black quantscale/ tests/ scripts/
   isort quantscale/ tests/ scripts/
   ```

6. **Commit your changes**:
   ```bash
   git add .
   git commit -m "feat: add new feature"
   ```

   Use conventional commit messages:
   - `feat:` New feature
   - `fix:` Bug fix
   - `docs:` Documentation changes
   - `test:` Test changes
   - `refactor:` Code refactoring
   - `perf:` Performance improvements
   - `chore:` Build/tooling changes

7. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

### Submitting the PR

1. Go to the original repository on GitHub
2. Click "New Pull Request"
3. Select your feature branch
4. Fill out the PR template with:
   - Description of changes
   - Related issues
   - Testing performed
   - Screenshots (if applicable)

### PR Review Process

- Maintainers will review your PR
- Address any requested changes
- Once approved, your PR will be merged

## Reporting Bugs

### Before Reporting

- Check existing [issues](https://github.com/tradervijeth/quantscale-distributed-training/issues)
- Verify the bug exists in the latest version
- Collect relevant information (error messages, environment details)

### Bug Report Template

Use the bug report issue template and include:

- **Description**: Clear description of the bug
- **Steps to Reproduce**: Detailed steps
- **Expected Behavior**: What should happen
- **Actual Behavior**: What actually happens
- **Environment**:
  - Python version
  - PyTorch version
  - CUDA version (if applicable)
  - Operating system
- **Code Sample**: Minimal reproducible example
- **Error Message**: Full error traceback

## Suggesting Enhancements

### Enhancement Proposal Template

Use the feature request template and include:

- **Problem**: What problem does this solve?
- **Solution**: Proposed solution
- **Alternatives**: Alternative solutions considered
- **Benefits**: How does this benefit users?
- **Implementation**: High-level implementation approach

## Development Guidelines

### Adding New Features

1. Discuss the feature in an issue first
2. Implement the feature with tests
3. Update documentation
4. Add examples if applicable
5. Update CHANGELOG.md

### Adding New Models

When adding a new model architecture:

1. Create a new file in `quantscale/models/`
2. Inherit from `nn.Module`
3. Include comprehensive docstrings
4. Add unit tests in `tests/test_models.py`
5. Add usage example in `examples/`
6. Update README.md

### Adding New Datasets

When adding a new dataset loader:

1. Create loader in `quantscale/data/loaders.py`
2. Add preprocessing in `quantscale/data/processors.py`
3. Add tests in `tests/test_data.py`
4. Document the dataset format

## Documentation

### Updating Documentation

- Keep README.md up to date
- Update docstrings for code changes
- Add examples for new features
- Update notebooks for significant changes

### Building Documentation

```bash
# Install Sphinx
pip install sphinx sphinx-rtd-theme

# Build documentation
cd docs
make html
```

## Release Process

(For maintainers)

1. Update version in `setup.py` and `quantscale/__init__.py`
2. Update CHANGELOG.md
3. Create git tag: `git tag -a v0.1.0 -m "Release v0.1.0"`
4. Push tag: `git push origin v0.1.0`
5. GitHub Actions will handle the release

## Questions?

If you have questions, feel free to:

- Open a [Discussion](https://github.com/tradervijeth/quantscale-distributed-training/discussions)
- Email: finance@vijeth.com

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.

---

**Thank you for contributing to QuantScale!** 🚀

Copyright © 2025 Vithushan Jeyapahan, Vijeth Ltd
