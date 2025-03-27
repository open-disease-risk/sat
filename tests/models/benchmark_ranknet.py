"""Benchmark script for RankNet loss implementations."""

import os
import tempfile
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

from sat.models.heads import SAOutput
from sat.loss.ranking.sample import SampleRankingLoss
from sat.loss.ranking.multievent import MultiEventRankingLoss
from sat.loss.ranking.sample_list_mle import SampleListMLELoss
from sat.loss.ranking.soap import SOAPLoss
from sat.loss.ranking.sample_soap import SampleSOAPLoss
from sat.loss.ranking.ranknet import RankNetLoss
from sat.loss.ranking.sample_ranknet import SampleRankNetLoss
from sat.loss.ranking.event_ranknet import EventRankNetLoss


def create_test_data(batch_size=32, num_events=2, num_time_bins=10):
    """Create synthetic test data for benchmarking."""
    # Create fake hazard and survival
    hazard = torch.rand(batch_size, num_events, num_time_bins, requires_grad=True)

    # Ensure survival is monotonically decreasing
    survival_base = 1.0 - torch.cumsum(torch.sigmoid(hazard), dim=2) / num_time_bins
    ones = torch.ones(batch_size, num_events, 1)
    survival = torch.cat([ones, survival_base], dim=2)

    # Create dummy logits
    logits = torch.zeros_like(hazard)

    # Create predictions
    predictions = SAOutput(logits=logits, hazard=hazard, survival=survival)

    # Create references tensor
    references = torch.zeros(batch_size, 4 * num_events)

    # Fill with synthetic event data
    for i in range(batch_size):
        event_idx = i % num_events
        # Set event indicator
        references[i, num_events + event_idx] = 1.0
        # Set duration
        references[i, 3 * num_events + event_idx] = float(i % num_time_bins)

    # Set some samples to have multiple events for competing risks
    if num_events > 1:
        for i in range(0, batch_size, 5):
            if i + 1 < batch_size:
                # Set all events to 1 for this sample
                references[i, num_events : 2 * num_events] = 1.0

                # Set different durations for each event
                for e in range(num_events):
                    references[i, 3 * num_events + e] = float((i + e) % num_time_bins)

    return predictions, references


def create_temp_files(num_time_bins=10, num_events=2):
    """Create temporary files needed for loss initialization."""
    # Create duration cuts file
    cuts_file = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
    for i in range(num_time_bins + 1):
        cuts_file.write(f"{float(i)}\n")
    cuts_file.close()

    # Create importance weights file
    weights_file = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
    # First weight is for censoring
    weights_file.write("1.0\n")
    # Remaining weights are for event types
    for _ in range(num_events):
        weights_file.write("0.5\n")
    weights_file.close()

    return cuts_file.name, weights_file.name


def run_benchmark(loss_fn, predictions, references, num_iterations=5, name="Unknown"):
    """Run performance benchmark for a given loss function."""
    # Ensure inputs require grad
    if isinstance(predictions.hazard, torch.Tensor):
        predictions.hazard.requires_grad_(True)

    # Warm-up run
    _ = loss_fn(predictions, references)

    # Time forward pass
    forward_times = []
    backward_times = []

    for _ in range(num_iterations):
        # Clean up gradients
        if predictions.hazard.grad is not None:
            predictions.hazard.grad.zero_()

        # Forward pass
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        loss = loss_fn(predictions, references)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()
        forward_times.append((end_time - start_time) * 1000)  # Convert to ms

        # Check if loss requires gradient
        if not loss.requires_grad:
            print(f"Warning: {name} does not produce gradients")
            backward_times.append(0.0)
            continue

        # Backward pass
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        loss.backward(
            retain_graph=True
        )  # Retain graph to allow multiple backward passes
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()
        backward_times.append((end_time - start_time) * 1000)  # Convert to ms

    # Calculate average times
    avg_forward = sum(forward_times) / len(forward_times)
    avg_backward = sum(backward_times) / len(backward_times)

    return avg_forward, avg_backward


def compare_ranking_losses(
    batch_sizes=[16, 32, 64, 128, 256], num_events=2, iterations=3
):
    """Compare different ranking loss implementations."""
    results = defaultdict(list)

    # Track metadata
    results["batch_size"] = []

    # Track loss names with prefixes for different metrics
    loss_names = [
        "sample_ranking",
        "multi_event",
        "sample_list_mle",
        "sample_soap",
        "sample_ranknet",
        "event_ranknet",
    ]

    for name in loss_names:
        results[f"{name}_forward"] = []
        results[f"{name}_backward"] = []
        results[f"{name}_total"] = []

    # Set up temp files for cuts and weights
    num_time_bins = 10
    duration_cuts, importance_weights = create_temp_files(num_time_bins, num_events)

    try:
        # Run benchmark for each batch size
        for batch_size in batch_sizes:
            print(f"Benchmarking batch_size={batch_size}, num_events={num_events}")

            # Create test data
            predictions, references = create_test_data(
                batch_size, num_events, num_time_bins
            )

            # Create loss functions
            sample_ranking = SampleRankingLoss(
                duration_cuts=duration_cuts,
                importance_sample_weights=importance_weights,
                num_events=num_events,
                sigma=1.0,
                margin=0.1,
            )

            multi_event = MultiEventRankingLoss(
                duration_cuts=duration_cuts,
                importance_sample_weights=importance_weights,
                num_events=num_events,
                sigma=1.0,
                margin=0.1,
            )

            sample_list_mle = SampleListMLELoss(
                duration_cuts=duration_cuts,
                importance_sample_weights=importance_weights,
                num_events=num_events,
                epsilon=1e-10,
                temperature=1.0,
            )

            sample_soap = SampleSOAPLoss(
                duration_cuts=duration_cuts,
                importance_sample_weights=importance_weights,
                num_events=num_events,
                margin=0.1,
                sigma=1.0,
                sampling_strategy="uniform",
            )

            sample_ranknet = SampleRankNetLoss(
                duration_cuts=duration_cuts,
                importance_sample_weights=importance_weights,
                num_events=num_events,
                sigma=1.0,
                sampling_ratio=0.3,
            )

            event_ranknet = EventRankNetLoss(
                duration_cuts=duration_cuts,
                importance_sample_weights=importance_weights,
                num_events=num_events,
                sigma=1.0,
                sampling_ratio=1.0,  # Use all pairs for event ranking
            )

            # Store batch size
            results["batch_size"].append(batch_size)

            # Run benchmarks
            loss_fns = {
                "sample_ranking": sample_ranking,
                "multi_event": multi_event,
                "sample_list_mle": sample_list_mle,
                "sample_soap": sample_soap,
                "sample_ranknet": sample_ranknet,
                "event_ranknet": event_ranknet,
            }

            # Store benchmark results
            for name, loss_fn in loss_fns.items():
                # Create fresh test data for each loss to avoid gradient accumulation issues
                fresh_preds, fresh_refs = create_test_data(
                    batch_size, num_events, num_time_bins
                )

                # Run benchmark
                forward_time, backward_time = run_benchmark(
                    loss_fn, fresh_preds, fresh_refs, iterations, name
                )

                # Store results
                results[f"{name}_forward"].append(forward_time)
                results[f"{name}_backward"].append(backward_time)
                results[f"{name}_total"].append(forward_time + backward_time)

            # Print summary for this batch size
            print("\nPerformance results:")
            for name in loss_names:
                forward = results[f"{name}_forward"][-1]
                backward = results[f"{name}_backward"][-1]
                total = results[f"{name}_total"][-1]
                print(
                    f"  {name:16} - Forward: {forward:.4f} ms, Backward: {backward:.4f} ms, Total: {total:.4f} ms"
                )

            # Print speedups relative to SampleRankingLoss
            baseline = results["sample_ranking_total"][-1]
            if baseline > 0:
                print("\nRelative speedup:")
                for name in loss_names:
                    if name != "sample_ranking":
                        total = results[f"{name}_total"][-1]
                        speedup = baseline / total if total > 0 else 0
                        print(f"  {name:16} - {speedup:.2f}x")

            print("\n" + "-" * 80)

        # Save results
        save_results(results, loss_names)

    finally:
        # Clean up temp files
        os.unlink(duration_cuts)
        os.unlink(importance_weights)


def save_results(results, loss_names):
    """Save benchmark results to CSV and generate plots."""
    # Create logs directory if it doesn't exist
    log_dir = os.path.join(os.getcwd(), "logs")
    os.makedirs(log_dir, exist_ok=True)

    # Convert to DataFrame for CSV
    rows = []
    batch_sizes = results["batch_size"]

    for i, batch_size in enumerate(batch_sizes):
        baseline_time = results["sample_ranking_total"][i]

        for name in loss_names:
            forward = results[f"{name}_forward"][i]
            backward = results[f"{name}_backward"][i]
            total = results[f"{name}_total"][i]

            # Calculate speedup relative to baseline
            speedup = baseline_time / total if total > 0 else 0

            row = {
                "batch_size": batch_size,
                "method": name,
                "forward_ms": forward,
                "backward_ms": backward,
                "total_ms": total,
                "speedup": speedup,
            }
            rows.append(row)

    # Create DataFrame and save to CSV
    df = pd.DataFrame(rows)
    csv_path = os.path.join(log_dir, "ranknet_benchmark.csv")
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")

    # Create plots
    plot_results(df, loss_names, log_dir)

    # Generate summary
    generate_summary(df, loss_names, log_dir)


def plot_results(df, loss_names, log_dir):
    """Create performance comparison plots."""
    # Set up plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Colors for different methods
    colors = plt.cm.tab10(np.linspace(0, 1, len(loss_names)))
    color_map = {name: colors[i] for i, name in enumerate(loss_names)}

    # Plot execution time
    for name in loss_names:
        method_df = df[df["method"] == name]
        ax1.plot(
            method_df["batch_size"],
            method_df["total_ms"],
            marker="o",
            linewidth=2,
            label=name,
            color=color_map[name],
        )

    ax1.set_title("Execution Time by Batch Size")
    ax1.set_xlabel("Batch Size")
    ax1.set_ylabel("Time (ms)")
    ax1.set_yscale("log")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot speedup relative to baseline
    for name in loss_names:
        if name != "sample_ranking":  # Skip baseline
            method_df = df[df["method"] == name]
            ax2.plot(
                method_df["batch_size"],
                method_df["speedup"],
                marker="o",
                linewidth=2,
                label=name,
                color=color_map[name],
            )

    # Add horizontal line at speedup = 1.0
    ax2.axhline(y=1.0, color="black", linestyle="--", alpha=0.5)

    ax2.set_title("Speedup vs. SampleRankingLoss")
    ax2.set_xlabel("Batch Size")
    ax2.set_ylabel("Speedup Factor")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plot_path = os.path.join(log_dir, "ranknet_comparison.png")
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")


def generate_summary(df, loss_names, log_dir):
    """Generate summary markdown document with key findings."""
    # Calculate average and max speedups
    speedups = {}
    for name in loss_names:
        if name != "sample_ranking":  # Skip baseline
            method_df = df[df["method"] == name]
            avg_speedup = method_df["speedup"].mean()
            max_speedup = method_df["speedup"].max()
            speedups[name] = (avg_speedup, max_speedup)

    # Find best method for each batch size
    best_by_size = {}
    for batch_size in df["batch_size"].unique():
        batch_df = df[
            (df["batch_size"] == batch_size) & (df["method"] != "sample_ranking")
        ]
        best_method = batch_df.loc[batch_df["speedup"].idxmax()]
        best_by_size[batch_size] = (best_method["method"], best_method["speedup"])

    # Generate markdown content
    markdown = "# RankNet Performance Comparison\n\n"
    markdown += "This document summarizes the performance of RankNet compared to other ranking losses.\n\n"

    # Create speedup table
    markdown += "## Speedup by Batch Size (vs. SampleRankingLoss)\n\n"
    markdown += "| Batch Size |"
    for name in loss_names:
        if name != "sample_ranking":
            markdown += f" {name} |"
    markdown += " Best Method |\n"

    markdown += "|" + "-" * 11 + "|"
    for name in loss_names:
        if name != "sample_ranking":
            markdown += "-" * len(name) + "----|"
    markdown += "-" * 15 + "|\n"

    for batch_size in sorted(df["batch_size"].unique()):
        markdown += f"| {batch_size} |"
        for name in loss_names:
            if name != "sample_ranking":
                method_speedup = df[
                    (df["batch_size"] == batch_size) & (df["method"] == name)
                ]["speedup"].values[0]
                markdown += f" {method_speedup:.2f}x |"

        best_method, best_speedup = best_by_size[batch_size]
        markdown += f" {best_method} ({best_speedup:.2f}x) |\n"

    # Key findings
    markdown += "\n## Key Findings\n\n"
    for name, (avg, max_val) in sorted(
        speedups.items(), key=lambda x: x[1][0], reverse=True
    ):
        markdown += f"- **{name}**: Average {avg:.2f}x speedup, Max {max_val:.2f}x\n"

    # Best overall method
    best_method = max(speedups.items(), key=lambda x: x[1][0])[0]
    avg, max_val = speedups[best_method]
    markdown += f"\n**Overall Best Performer**: {best_method} (Average: {avg:.2f}x)\n\n"

    # Method characteristics
    markdown += "## Method Characteristics\n\n"

    markdown += "### SampleRankingLoss (Baseline)\n"
    markdown += "- Traditional pairwise ranking approach using margin-based loss\n"
    markdown += "- O(n²) complexity with batch size\n"
    markdown += "- Complete pairwise comparison of all samples\n\n"

    markdown += "### MultiEventRankingLoss\n"
    markdown += "- Compares different event types within the same observation\n"
    markdown += "- O(e²) complexity with number of events\n"
    markdown += "- Margin-based pairwise comparisons\n\n"

    markdown += "### SampleListMLELoss\n"
    markdown += "- Listwise ranking approach using Plackett-Luce model\n"
    markdown += "- O(n log n) complexity\n"
    markdown += "- Directly optimizes probability of correct ordering\n\n"

    markdown += "### SampleSOAPLoss\n"
    markdown += "- Accelerated pairwise approach with strategic pair sampling\n"
    markdown += "- Reduces complexity to approximately O(n log n)\n"
    markdown += "- Uses margin-based comparisons with optimized implementation\n\n"

    markdown += "### SampleRankNetLoss\n"
    markdown += "- Probabilistic pairwise ranking using logistic function\n"
    markdown += "- Smooth differentiable loss with cross-entropy\n"
    markdown += "- Adaptive sampling of pairs to reduce computation\n\n"

    markdown += "### EventRankNetLoss\n"
    markdown += "- RankNet approach for comparing different event types\n"
    markdown += "- Probabilistic ranking between competing risks\n"
    markdown += "- Particularly effective for multi-event scenarios\n\n"

    # Save markdown
    md_path = os.path.join(log_dir, "ranknet_comparison.md")
    with open(md_path, "w") as f:
        f.write(markdown)
    print(f"Summary saved to {md_path}")


if __name__ == "__main__":
    # Run benchmark
    compare_ranking_losses(
        batch_sizes=[16, 32, 64, 128, 256], num_events=2, iterations=3
    )
