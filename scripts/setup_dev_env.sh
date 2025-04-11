#!/bin/bash
# Script to set up the development environment for SAT

# Ensure we exit on any error
set -e

echo "Setting up development environment for SAT..."

# Check if poetry is installed
if ! command -v poetry &> /dev/null; then
    echo "Poetry not found. Please install it first:"
    echo "pip install poetry"
    exit 1
fi

# Check if pre-commit is installed
if ! command -v pre-commit &> /dev/null; then
    echo "Installing pre-commit..."
    pip install pre-commit
fi

# Install dependencies
echo "Installing project dependencies..."
poetry install

# Install pre-commit hooks
echo "Installing pre-commit hooks..."
pre-commit install

# Run pre-commit on all files to verify
echo "Running pre-commit on all files to verify setup..."
pre-commit run --all-files || {
    echo "Note: Some pre-commit hooks failed. This is expected for the first run as they may automatically fix issues."
    echo "You can commit the changes made by pre-commit, then run again to verify."
}

echo "Development environment setup complete!"
echo "See docs/development_guide.md for more information on development workflow."