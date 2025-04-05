"""
Benchmark the impact of MoCo buffer size on training performance and stability.

This script tests how different buffer sizes affect:
1. Loss stability (variance)
2. Training time
3. Memory usage
4. Final model performance

Usage:
    poetry run python -m tests.models.benchmark_moco
"""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from transformers.utils import ModelOutput

from sat.loss import SATNLLPCHazardLoss
from sat.loss.momentum_buffer import MoCoSurvivalLoss, MomentumBuffer
from sat.utils import logging

logger = logging.get_default_logger()


def mock_data(
    batch_size: int,
    num_events: int = 1,
    censoring_rate: float = 0.7,
    embedding_dim: int = 128,
    num_cuts: int = 10,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """
    Create mock data for benchmarking.

    Args:
        batch_size: Number of samples in batch
        num_events: Number of events
        censoring_rate: Proportion of censored samples
        embedding_dim: Dimension of embeddings
        num_cuts: Number of time discretization cuts
        device: Device to place tensors on

    Returns:
        Dictionary of mock data tensors
    """
    # Generate random embeddings
    logits = torch.randn(batch_size, num_cuts, device=device)
    logits.requires_grad = True

    # Create references tensor [batch_size x 4*num_events]
    references = torch.zeros(batch_size, 4 * num_events, device=device)

    # Duration percentiles (discretized)
    references[:, 0:num_events] = torch.randint(
        0, num_cuts, (batch_size, num_events), device=device
    )

    # Event indicators (with censoring)
    num_uncensored = int(batch_size * (1 - censoring_rate))
    uncensored_indices = torch.randperm(batch_size)[:num_uncensored]
    references[uncensored_indices, num_events : 2 * num_events] = 1

    # Fraction with quantile
    references[:, 2 * num_events : 3 * num_events] = torch.rand(
        batch_size, num_events, device=device
    )

    # Durations
    references[:, 3 * num_events : 4 * num_events] = (
        torch.rand(batch_size, num_events, device=device) * 10.0
    )  # Random durations between 0 and 10

    # Create mock output
    predictions = ModelOutput(logits=logits)

    return {"predictions": predictions, "references": references}


def benchmark_buffer_sizes(
    buffer_sizes: List[int],
    batch_size: int = 32,
    num_events: int = 1,
    censoring_rate: float = 0.7,
    embedding_dim: int = 128,
    num_cuts: int = 10,
    num_iterations: int = 100,
    device: str = "cpu",
) -> pd.DataFrame:
    """
    Benchmark performance across different buffer sizes.

    Args:
        buffer_sizes: List of buffer sizes to test
        batch_size: Batch size to use
        num_events: Number of events
        censoring_rate: Proportion of censored samples
        embedding_dim: Dimension of embeddings
        num_cuts: Number of time discretization cuts
        num_iterations: Number of training iterations
        device: Device to run benchmark on

    Returns:
        DataFrame with benchmark results
    """
    results = []

    for buffer_size in buffer_sizes:
        logger.info(f"Benchmarking buffer size: {buffer_size}")

        # Create base loss
        base_loss = SATNLLPCHazardLoss(
            num_events=num_events,
            duration_cuts=None,  # Not used in mock benchmark
        )

        # Register buffer for cuts
        base_loss.register_buffer("duration_cuts", torch.linspace(0, 10, num_cuts))

        # Create MoCo loss
        moco_loss = MoCoSurvivalLoss(
            base_loss=base_loss,
            buffer_size=buffer_size,
            num_events=num_events,
            embedding_dim=embedding_dim,
            use_buffer=True,
            current_batch_weight=1.0,
            buffer_weight=1.0,
        )

        moco_loss.train()  # Set to training mode
        moco_loss.to(device)

        # Track metrics
        loss_values = []
        forward_times = []
        backward_times = []
        memory_usage = []

        # Create optimizer
        dummy_params = torch.nn.Parameter(
            torch.randn(1, requires_grad=True, device=device)
        )
        optimizer = torch.optim.Adam([dummy_params], lr=0.001)

        # Run iterations
        for i in range(num_iterations):
            # Create mock data
            data = mock_data(
                batch_size=batch_size,
                num_events=num_events,
                censoring_rate=censoring_rate,
                embedding_dim=embedding_dim,
                num_cuts=num_cuts,
                device=device,
            )

            # Measure forward pass time
            start_time = time.time()
            loss_val = moco_loss(data["predictions"], data["references"])
            forward_time = time.time() - start_time

            # Measure backward pass time
            start_time = time.time()
            loss_val.backward()
            optimizer.step()
            optimizer.zero_grad()
            backward_time = time.time() - start_time

            # Record metrics
            loss_values.append(loss_val.item())
            forward_times.append(forward_time)
            backward_times.append(backward_time)

            # Measure memory usage on CUDA
            if device.startswith("cuda"):
                memory_usage.append(torch.cuda.max_memory_allocated() / (1024**2))  # MB
            else:
                memory_usage.append(0)  # Not tracking memory on CPU

            if i % 10 == 0:
                logger.info(f"  Iteration {i}: loss={loss_val.item():.4f}")

        # Calculate metrics
        buffer_stats = moco_loss.get_buffer_stats()

        # Calculate loss variance over windows of 10 iterations
        window_size = min(10, len(loss_values))
        variances = []
        for i in range(len(loss_values) - window_size + 1):
            window = loss_values[i : i + window_size]
            variances.append(np.var(window))

        mean_variance = np.mean(variances) if variances else 0

        # Store results
        results.append(
            {
                "buffer_size": buffer_size,
                "mean_loss": np.mean(loss_values),
                "loss_variance": mean_variance,
                "loss_variance_pct": (
                    mean_variance / np.mean(loss_values) * 100
                    if np.mean(loss_values) > 0
                    else 0
                ),
                "max_loss": np.max(loss_values),
                "min_loss": np.min(loss_values),
                "mean_forward_time": np.mean(forward_times),
                "mean_backward_time": np.mean(backward_times),
                "total_time": np.sum(forward_times) + np.sum(backward_times),
                "mean_memory_mb": (
                    np.mean(memory_usage) if memory_usage[0] > 0 else None
                ),
                "max_memory_mb": np.max(memory_usage) if memory_usage[0] > 0 else None,
                "buffer_utilization": buffer_stats.get("buffer_utilization", 0),
            }
        )

        # Free memory
        moco_loss.reset_buffer()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    return results_df


def plot_results(results_df: pd.DataFrame, output_dir: Optional[str] = None):
    """
    Plot benchmark results.

    Args:
        results_df: DataFrame with benchmark results
        output_dir: Directory to save plots in
    """
    plt.figure(figsize=(15, 10))

    # Plot 1: Loss variance vs buffer size
    plt.subplot(2, 2, 1)
    plt.plot(results_df["buffer_size"], results_df["loss_variance_pct"], "o-")
    plt.xlabel("Buffer Size")
    plt.ylabel("Loss Variance (% of mean)")
    plt.title("Loss Variance vs Buffer Size")
    plt.grid(True)

    # Plot 2: Total time vs buffer size
    plt.subplot(2, 2, 2)
    plt.plot(results_df["buffer_size"], results_df["total_time"], "o-")
    plt.xlabel("Buffer Size")
    plt.ylabel("Total Time (seconds)")
    plt.title("Training Time vs Buffer Size")
    plt.grid(True)

    # Plot 3: Memory usage vs buffer size
    plt.subplot(2, 2, 3)
    if (
        "max_memory_mb" in results_df
        and results_df["max_memory_mb"].iloc[0] is not None
    ):
        plt.plot(results_df["buffer_size"], results_df["max_memory_mb"], "o-")
        plt.xlabel("Buffer Size")
        plt.ylabel("Max Memory Usage (MB)")
        plt.title("Memory Usage vs Buffer Size")
        plt.grid(True)
    else:
        plt.text(
            0.5,
            0.5,
            "Memory usage data not available",
            horizontalalignment="center",
            verticalalignment="center",
        )
        plt.title("Memory Usage (Not Available)")

    # Plot 4: Forward/backward time vs buffer size
    plt.subplot(2, 2, 4)
    plt.plot(
        results_df["buffer_size"],
        results_df["mean_forward_time"],
        "o-",
        label="Forward",
    )
    plt.plot(
        results_df["buffer_size"],
        results_df["mean_backward_time"],
        "o-",
        label="Backward",
    )
    plt.xlabel("Buffer Size")
    plt.ylabel("Time (seconds)")
    plt.title("Forward/Backward Time vs Buffer Size")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    if output_dir:
        output_path = Path(output_dir) / "moco_benchmark_results.png"
        plt.savefig(output_path)
        logger.info(f"Saved plot to {output_path}")

    plt.show()


def benchmark_optimal_buffer_estimation(
    dataset_sizes: List[int],
    censoring_rates: List[float],
    batch_sizes: List[int],
    min_events_targets: List[int],
) -> pd.DataFrame:
    """
    Benchmark the optimal buffer size estimator.

    Args:
        dataset_sizes: List of dataset sizes
        censoring_rates: List of censoring rates
        batch_sizes: List of batch sizes
        min_events_targets: List of minimum event targets

    Returns:
        DataFrame with estimation results
    """
    results = []

    for dataset_size in dataset_sizes:
        for censoring_rate in censoring_rates:
            for batch_size in batch_sizes:
                for min_events in min_events_targets:
                    # Estimate optimal buffer size
                    buffer_size = MomentumBuffer.estimate_optimal_buffer_size(
                        num_samples=dataset_size,
                        censoring_rate=censoring_rate,
                        min_events_per_batch=min_events,
                        batch_size=batch_size,
                    )

                    # Calculate events per batch
                    events_per_batch = batch_size * (1 - censoring_rate)

                    # Calculate effective events with buffer
                    effective_events = events_per_batch * (1 + buffer_size / batch_size)

                    results.append(
                        {
                            "dataset_size": dataset_size,
                            "censoring_rate": censoring_rate,
                            "batch_size": batch_size,
                            "min_events_target": min_events,
                            "estimated_buffer_size": buffer_size,
                            "events_per_batch": events_per_batch,
                            "effective_events": effective_events,
                            "buffer_to_batch_ratio": buffer_size / batch_size,
                        }
                    )

    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    return results_df


def benchmark_variance_tracking(
    batch_size: int = 32,
    num_events: int = 1,
    censoring_rate: float = 0.7,
    embedding_dim: int = 128,
    num_cuts: int = 10,
    buffer_size: int = 1024,
    initial_buffer_size: int = 128,
    num_iterations: int = 200,
    device: str = "cpu",
    noise_episodes: Optional[List[int]] = None,
) -> Dict:
    """
    Benchmark the variance tracking and adaptive buffer size adjustment.

    Args:
        batch_size: Batch size to use
        num_events: Number of events
        censoring_rate: Proportion of censored samples
        embedding_dim: Dimension of embeddings
        num_cuts: Number of time discretization cuts
        buffer_size: Maximum buffer size
        initial_buffer_size: Initial buffer size
        num_iterations: Number of training iterations
        device: Device to run benchmark on
        noise_episodes: Iterations at which to add noise to simulate instability

    Returns:
        Dictionary with tracking results
    """
    # Default noise episodes if not provided
    if noise_episodes is None:
        noise_episodes = [50, 100, 150]

    # Create base loss
    base_loss = SATNLLPCHazardLoss(
        num_events=num_events,
        duration_cuts=None,  # Not used in mock benchmark
    )

    # Register buffer for cuts
    base_loss.register_buffer("duration_cuts", torch.linspace(0, 10, num_cuts))

    # Create MoCo loss with adaptive buffer
    moco_loss = MoCoSurvivalLoss(
        base_loss=base_loss,
        buffer_size=buffer_size,
        num_events=num_events,
        embedding_dim=embedding_dim,
        use_buffer=True,
        current_batch_weight=1.0,
        buffer_weight=1.0,
        dynamic_buffer=True,
        initial_buffer_size=initial_buffer_size,
        track_variance=True,
        adaptive_buffer=True,
    )

    moco_loss.train()  # Set to training mode
    moco_loss.to(device)

    # Track metrics
    loss_values = []
    buffer_sizes = []
    variance_history = []
    adjustment_history = []

    # Create optimizer
    dummy_params = torch.nn.Parameter(torch.randn(1, requires_grad=True, device=device))
    optimizer = torch.optim.Adam([dummy_params], lr=0.001)

    # Run iterations
    for i in range(num_iterations):
        # Create mock data
        data = mock_data(
            batch_size=batch_size,
            num_events=num_events,
            censoring_rate=censoring_rate,
            embedding_dim=embedding_dim,
            num_cuts=num_cuts,
            device=device,
        )

        # Add artificial noise during noise episodes to test adaptation
        if i in noise_episodes:
            logger.info(f"Adding noise at iteration {i}")
            # Create noisy gradients by using a much higher learning rate temporarily
            optimizer.param_groups[0]["lr"] = 0.1
        elif i in [ep + 1 for ep in noise_episodes]:
            # Restore normal learning rate
            optimizer.param_groups[0]["lr"] = 0.001

        # Forward pass
        loss_val = moco_loss(data["predictions"], data["references"])

        # Backward pass
        loss_val.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Record metrics
        loss_values.append(loss_val.item())
        buffer_sizes.append(moco_loss.buffer.current_buffer_size)

        # Get variance history
        variance_history = moco_loss.buffer.loss_variance_history.copy()
        adjustment_history = moco_loss.buffer.buffer_adjustment_history.copy()

        if i % 10 == 0:
            buffer_stats = moco_loss.get_buffer_stats()
            logger.info(
                f"Iteration {i}: loss={loss_val.item():.4f}, "
                f"buffer_size={moco_loss.buffer.current_buffer_size}, "
                f"adjustments={len(adjustment_history)}"
            )

    # Collect results
    results = {
        "loss_values": loss_values,
        "buffer_sizes": buffer_sizes,
        "variance_history": variance_history,
        "adjustment_history": adjustment_history,
        "iterations": list(range(num_iterations)),
        "noise_episodes": noise_episodes,
    }

    return results


def plot_variance_tracking_results(results: Dict, output_dir: Optional[str] = None):
    """
    Plot variance tracking benchmark results.

    Args:
        results: Dictionary with tracking results
        output_dir: Directory to save plots in
    """
    plt.figure(figsize=(15, 10))

    # Plot 1: Loss values over time
    plt.subplot(3, 1, 1)
    plt.plot(results["iterations"], results["loss_values"])

    # Mark noise episodes
    for ep in results["noise_episodes"]:
        plt.axvline(x=ep, color="r", linestyle="--", alpha=0.5)

    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Loss Values Over Time")
    plt.grid(True)

    # Plot 2: Buffer size over time
    plt.subplot(3, 1, 2)
    plt.plot(results["iterations"], results["buffer_sizes"])

    # Mark noise episodes
    for ep in results["noise_episodes"]:
        plt.axvline(x=ep, color="r", linestyle="--", alpha=0.5)

    # Mark buffer adjustments
    for adj in results["adjustment_history"]:
        it_index = adj[0]  # First element is variance history index
        plt.scatter(
            [it_index],
            [adj[2]],
            color="g" if adj[3] == "increase" else "b",
            marker="^" if adj[3] == "increase" else "v",
            s=100,
        )

    plt.xlabel("Iteration")
    plt.ylabel("Buffer Size")
    plt.title("Buffer Size Adaptation Over Time")
    plt.grid(True)

    # Plot 3: Loss variance over time
    plt.subplot(3, 1, 3)
    iterations = list(range(len(results["variance_history"])))
    plt.plot(iterations, results["variance_history"])

    # Mark buffer adjustments
    for adj in results["adjustment_history"]:
        it_index = adj[0]  # First element is variance history index
        variance = adj[4]  # Last element is variance at adjustment
        plt.scatter(
            [it_index],
            [variance],
            color="g" if adj[3] == "increase" else "b",
            marker="^" if adj[3] == "increase" else "v",
            s=100,
        )

    plt.xlabel("Variance Measurement Index")
    plt.ylabel("Loss Variance")
    plt.title("Loss Variance Over Time")
    plt.grid(True)

    plt.tight_layout()

    if output_dir:
        output_path = Path(output_dir) / "moco_variance_tracking_results.png"
        plt.savefig(output_path)
        logger.info(f"Saved plot to {output_path}")

    plt.show()


def main():
    """Main benchmark entry point."""
    parser = argparse.ArgumentParser(description="Benchmark MoCo implementations")
    parser.add_argument(
        "--benchmark",
        type=str,
        choices=["buffer", "estimation", "variance"],
        default="buffer",
        help="Benchmark type to run",
    )
    parser.add_argument(
        "--buffer-sizes",
        type=int,
        nargs="+",
        default=[0, 32, 64, 128, 256, 512, 1024, 2048],
        help="Buffer sizes to benchmark",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--num-events", type=int, default=1, help="Number of events")
    parser.add_argument(
        "--censoring-rate",
        type=float,
        default=0.7,
        help="Censoring rate (proportion of censored samples)",
    )
    parser.add_argument(
        "--num-iterations", type=int, default=100, help="Number of training iterations"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./logs", help="Directory to save results"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run benchmark on (cpu or cuda)",
    )

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run selected benchmark
    if args.benchmark == "buffer":
        logger.info("Running buffer size benchmark...")
        results_df = benchmark_buffer_sizes(
            buffer_sizes=args.buffer_sizes,
            batch_size=args.batch_size,
            num_events=args.num_events,
            censoring_rate=args.censoring_rate,
            num_iterations=args.num_iterations,
            device=args.device,
        )

        # Save results
        results_path = output_dir / "moco_benchmark_results.csv"
        results_df.to_csv(results_path, index=False)
        logger.info(f"Saved results to {results_path}")

        # Plot results
        plot_results(results_df, output_dir=str(output_dir))

    elif args.benchmark == "estimation":
        logger.info("Running buffer size estimation benchmark...")
        results_df = benchmark_optimal_buffer_estimation(
            dataset_sizes=[1000, 5000, 10000],
            censoring_rates=[0.5, 0.7, 0.9],
            batch_sizes=[16, 32, 64],
            min_events_targets=[5, 10, 20],
        )

        # Save results
        results_path = output_dir / "moco_estimation_results.csv"
        results_df.to_csv(results_path, index=False)
        logger.info(f"Saved results to {results_path}")

        # Print summary
        print(
            results_df.groupby(["censoring_rate", "min_events_target"])[
                "buffer_to_batch_ratio"
            ].mean()
        )

    elif args.benchmark == "variance":
        logger.info("Running variance tracking benchmark...")
        results = benchmark_variance_tracking(
            batch_size=args.batch_size,
            num_events=args.num_events,
            censoring_rate=args.censoring_rate,
            buffer_size=2048,
            initial_buffer_size=128,
            num_iterations=args.num_iterations * 2,  # Longer run to see adaptations
            device=args.device,
        )

        # Plot results
        plot_variance_tracking_results(results, output_dir=str(output_dir))


if __name__ == "__main__":
    main()
