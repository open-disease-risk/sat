# SAT (Survival Analysis using Transformers) CLI Reference

## Build & Test Commands
```
# Install project dependencies 
poetry install

# Activate environment
poetry shell

# Run all tests
python -m pytest tests/

# Run a specific test file
python -m pytest tests/utils/test_output.py

# Run a specific test class or method
python -m pytest tests/utils/test_output.py::TestLogMetricsFromReplications
python -m pytest tests/utils/test_output.py::TestLogMetricsFromReplications::test_single_value

# Run the finetuning script (example)
python -m sat.finetune experiments=seer/survival
```

## Code Style Guidelines
- **Imports**: Standard library first, third-party packages second, local modules last
- **Classes**: Use Pascal case (TestLogMetricsFromReplications)
- **Functions/Variables**: Use snake_case (test_single_value)
- **Constants**: Use UPPER_SNAKE_CASE
- **Typing**: Add type hints for function parameters and return values
- **Testing**: Use unittest framework with descriptive test method names
- **Docstrings**: Triple quotes for module and function documentation
- **Error Handling**: Use appropriate exception types and descriptive messages