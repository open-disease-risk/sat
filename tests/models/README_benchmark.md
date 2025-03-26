# Ranking Loss Optimization Benchmarks

This directory contains benchmarking scripts to evaluate the performance of the original and optimized ranking loss implementations.

## Running the Benchmarks

To run the benchmarks, use the `benchmark_optimized_ranking.py` script. This script compares the performance of different ranking loss implementations across various batch sizes and event counts.

### Prerequisites

Make sure you have all required packages installed:

```bash
# Install tabulate package for generating markdown tables
poetry add --group test tabulate
```

### Basic Usage

```bash
# Run with default settings
poetry run python tests/models/benchmark_optimized_ranking.py

# Run with custom batch sizes and event counts
poetry run python tests/models/benchmark_optimized_ranking.py --batch-sizes 64 128 256 --event-counts 1 2 --num-trials 3
```

### Command Line Arguments

The script supports the following command line arguments:

- `--batch-sizes`: List of batch sizes to test (default: [32, 128, 512])
- `--event-counts`: List of event counts to test (default: [1, 2, 5])
- `--num-trials`: Number of trials per configuration (default: 5)
- `--device`: Device to run on ("auto", "cpu", "cuda", or "mps") (default: "auto")
- `--output-dir`: Directory to save results (default: "logs")

### Hardware Acceleration

The script automatically selects the best available device (CUDA GPU, Apple MPS, or CPU) when using the default "auto" device option.

For specific device selection:

```bash
# Run on CPU
poetry run python tests/models/benchmark_optimized_ranking.py --device cpu

# Run on CUDA GPU (if available)
poetry run python tests/models/benchmark_optimized_ranking.py --device cuda

# Run on Apple Metal (if available)
poetry run python tests/models/benchmark_optimized_ranking.py --device mps
```

## Output

The benchmark generates several outputs:

1. Benchmark results in CSV format
2. Summary tables in CSV format
3. A markdown report with tables summarizing the results
4. Visualizations comparing the performance of the implementations

All outputs are saved to the specified output directory (default: "logs").

## Benchmark Methodology

The benchmark:

1. Creates synthetic survival data with specified dimensions
2. Runs the original and optimized implementations of ranking losses
3. Measures forward pass, backward pass, and total execution time
4. Calculates speedup factors between implementations
5. Generates visualizations and summary tables

## Example

Running with 2 event types and batch sizes of 64 and 128:

```bash
poetry run python tests/models/benchmark_optimized_ranking.py --batch-sizes 64 128 --event-counts 2 --num-trials 3
```

This will run the benchmark for:
- SampleRankingLoss (original implementation)
- SampleRankingLoss (optimized implementation)
- MultiEventRankingLoss (original implementation)
- MultiEventRankingLoss (optimized implementation)

with the specified configurations.