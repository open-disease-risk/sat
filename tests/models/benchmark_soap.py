"""Benchmark SOAP loss against other ranking losses for survival analysis."""

import os
import tempfile
import pandas as pd
import torch
import time
import numpy as np
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt
from collections import defaultdict

from sat.models.heads import SAOutput
from sat.loss.ranking.multievent import MultiEventRankingLoss
from sat.loss.ranking.sample import SampleRankingLoss
from sat.loss.ranking.sample_list_mle import SampleListMLELoss
from sat.loss.ranking.survrnc import SurvRNCLoss
from sat.loss.ranking.sample_soap import SampleSOAPLoss
from sat.loss.base import RankingLoss


def create_fake_data(
    batch_size: int = 16, num_events: int = 2, num_cuts: int = 10
) -> Tuple[SAOutput, torch.Tensor]:
    """Create synthetic data for testing ranking losses."""
    # Create fake hazard and survival tensors
    hazard = torch.rand(batch_size, num_events, num_cuts)

    # Ensure survival is decreasing for visualization
    survival_base = torch.cumsum(
        torch.nn.functional.softplus(torch.randn(batch_size, num_events, num_cuts)),
        dim=2,
    )
    # Scale to 0-1 range and flip to get decreasing values
    max_vals = survival_base.max(dim=2, keepdim=True)[0]
    survival_base = 1 - (survival_base / (max_vals + 1e-6))
    # Add survival at time 0 (always 1.0)
    ones = torch.ones(batch_size, num_events, 1)
    survival = torch.cat([ones, survival_base], dim=2)

    # Create logits (we don't need them for the test)
    logits = torch.zeros(batch_size, num_events, num_cuts)

    # Create targets
    # Each row is: [duration_percentile, event, fraction, duration] for each event
    # For num_events=2, shape will be [batch_size, 8]
    targets = torch.zeros(batch_size, 4 * num_events)

    # Set some events to 1
    # Ensure durations are within the range of our cuts (1-100)
    for i in range(batch_size):
        event_type = i % num_events
        targets[i, num_events + event_type] = 1  # Set event indicator
        # Scale durations to be within the range 1-90 (below our max cut point)
        targets[i, 3 * num_events + event_type] = 1 + (i % 90)  # Set duration within range
        # Set duration index (percentile)
        targets[i, event_type] = i % num_cuts

    # For competing risks test, make some samples have multiple events
    # Only do this if we have more than one event type
    if num_events > 1:
        for i in range(0, batch_size, 5):
            if i + 1 < batch_size:
                # Set both events to 1 for this sample
                targets[i, num_events : 2 * num_events] = 1.0
                # Set different durations for each event (ensuring they're in range)
                targets[i, 3 * num_events] = 1 + (i % 80)  # First event duration 
                targets[i, 3 * num_events + 1] = 10 + (i % 80)  # Second event duration

    # Create fake predictions
    predictions = SAOutput(logits=logits, hazard=hazard, survival=survival)

    return predictions, targets


def create_duration_cuts_file(num_cuts: int = 10) -> str:
    """Create a temporary file with duration cuts."""
    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        # Start from 0 to ensure all durations are covered
        duration_cuts = np.linspace(0, 100, num_cuts)
        for cut in duration_cuts:
            f.write(f"{cut}\n")
        return f.name


def create_importance_weights_file(num_events: int = 2) -> str:
    """Create a temporary file with importance weights."""
    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        weights = np.ones(num_events + 1)  # Add one for censoring weight
        for weight in weights:
            f.write(f"{weight}\n")
        return f.name


def run_loss_benchmark(loss_fn, predictions, targets, num_iterations=3, name="Unknown"):
    """Run benchmark for a specific loss function."""
    forward_times = []
    backward_times = []
    
    for _ in range(num_iterations):
        try:
            # Forward pass
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.time()
            loss = loss_fn(predictions, targets)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.time()
            forward_times.append(end_time - start_time)
            
            # Check if loss requires grad
            if not loss.requires_grad:
                print(f"Warning: {name} loss does not require gradient")
                backward_times.append(0.0)
                continue
                
            # Backward pass
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.time()
            loss.backward()
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.time()
            backward_times.append(end_time - start_time)
            
        except Exception as e:
            print(f"Error with {name}: {e}")
            forward_times.append(0.0)
            backward_times.append(0.0)
    
    # Calculate average times (in ms)
    avg_forward = np.mean(forward_times) * 1000 if forward_times else 0
    avg_backward = np.mean(backward_times) * 1000 if backward_times else 0
    
    return avg_forward, avg_backward


def benchmark_all_losses(batch_sizes=[8, 16, 32, 64, 128], num_events_list=[1, 2], num_cuts=10, num_iterations=3):
    """Benchmark all ranking losses for different batch sizes and event counts."""
    # Results dictionary
    results = defaultdict(list)
    loss_names = [
        "sample_ranking", 
        "multi_event", 
        "sample_list_mle", 
        "survrnc",
        "sample_soap_uniform",
        "sample_soap_importance",
    ]
    
    # Create standard keys for all metrics
    for name in loss_names:
        results[f"{name}_forward"] = []
        results[f"{name}_backward"] = []
        results[f"{name}_total"] = []
    
    # Add metadata
    results["batch_size"] = []
    results["num_events"] = []
    
    # Run benchmarks for each configuration
    for batch_size in batch_sizes:
        for num_events in num_events_list:
            print(f"Benchmarking batch_size={batch_size}, num_events={num_events}")
            
            # Create files needed for loss initialization
            duration_cuts_file = create_duration_cuts_file(num_cuts)
            importance_weights_file = create_importance_weights_file(num_events)
            
            try:
                # Create loss instances
                sample_ranking_loss = SampleRankingLoss(
                    duration_cuts=duration_cuts_file,
                    importance_sample_weights=importance_weights_file,
                    num_events=num_events,
                    sigma=0.1,
                    margin=0.05,
                )
                
                multi_event_loss = MultiEventRankingLoss(
                    duration_cuts=duration_cuts_file,
                    importance_sample_weights=importance_weights_file,
                    num_events=num_events,
                    sigma=0.1,
                    margin=0.05,
                )
                
                sample_list_mle_loss = SampleListMLELoss(
                    duration_cuts=duration_cuts_file,
                    importance_sample_weights=importance_weights_file,
                    num_events=num_events,
                    epsilon=1e-10,
                    temperature=1.0,
                )
                
                survrnc_loss = SurvRNCLoss(
                    duration_cuts=duration_cuts_file,
                    importance_sample_weights=importance_weights_file,
                    num_events=num_events,
                    margin=0.5,
                    temperature=0.1,
                    use_hard_mining=(batch_size > 128),
                    mining_ratio=0.5,
                )
                
                sample_soap_uniform_loss = SampleSOAPLoss(
                    duration_cuts=duration_cuts_file,
                    importance_sample_weights=importance_weights_file,
                    num_events=num_events,
                    margin=0.1,
                    sigma=1.0,
                    num_pairs=None,  # Auto-calculate
                    sampling_strategy="uniform",
                    adaptive_margin=False,
                )

                sample_soap_importance_loss = SampleSOAPLoss(
                    duration_cuts=duration_cuts_file,
                    importance_sample_weights=importance_weights_file,
                    num_events=num_events,
                    margin=0.1,
                    sigma=1.0,
                    num_pairs=None,  # Auto-calculate
                    sampling_strategy="importance",
                    adaptive_margin=True,
                )
                
                # Create test data with gradients
                predictions, targets = create_fake_data(batch_size, num_events, num_cuts)
                
                # Run benchmarks for each loss function
                loss_instances = {
                    "sample_ranking": sample_ranking_loss,
                    "multi_event": multi_event_loss,
                    "sample_list_mle": sample_list_mle_loss,
                    "survrnc": survrnc_loss,
                    "sample_soap_uniform": sample_soap_uniform_loss,
                    "sample_soap_importance": sample_soap_importance_loss,
                }
                
                # Store metadata for this configuration
                results["batch_size"].append(batch_size)
                results["num_events"].append(num_events)
                
                # Benchmark all loss functions
                for name, loss_fn in loss_instances.items():
                    # Reset gradients between loss functions
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # Create fresh predictions tensor with gradients for each loss function
                    fresh_hazard = torch.rand(
                        batch_size, num_events, num_cuts, requires_grad=True
                    )
                    ones = torch.ones(batch_size, num_events, 1)
                    fresh_survival_base = (
                        1 - torch.cumsum(torch.nn.functional.softplus(fresh_hazard), dim=2) / num_cuts
                    )
                    fresh_survival = torch.cat([ones, fresh_survival_base], dim=2)
                    fresh_logits = torch.zeros_like(fresh_hazard)
                    fresh_predictions = SAOutput(
                        logits=fresh_logits, 
                        hazard=fresh_hazard, 
                        survival=fresh_survival
                    )
                    
                    # Debugging for SOAP losses
                    if "soap" in name:
                        try:
                            # Just call forward once for debugging
                            print(f"DEBUG: Testing {name} forward pass...")
                            
                            # Create test data with explicit event patterns
                            small_batch = 8  # Small batch for debugging
                            test_refs = torch.zeros(small_batch, 4 * num_events, requires_grad=False)
                            
                            # Set event indicators (1 or 0)
                            for i in range(small_batch):
                                if i % 2 == 0:  # Even samples have event 0
                                    test_refs[i, num_events] = 1
                                    test_refs[i, 3 * num_events] = float(i + 1)  # duration
                                else:  # Odd samples have event 1 (if available)
                                    if num_events > 1:
                                        test_refs[i, num_events + 1] = 1
                                        test_refs[i, 3 * num_events + 1] = float(i + 1)  # duration
                                    else:
                                        test_refs[i, num_events] = 1
                                        test_refs[i, 3 * num_events] = float(i + 1)  # duration
                            
                            # Create simple survival curves with small batch
                            test_hazard = torch.rand(small_batch, num_events, num_cuts, requires_grad=True)
                            ones = torch.ones(small_batch, num_events, 1)
                            test_surv_base = 1 - torch.cumsum(torch.sigmoid(test_hazard), dim=2) / num_cuts
                            test_surv = torch.cat([ones, test_surv_base], dim=2)
                            test_logits = torch.zeros_like(test_hazard)
                            test_preds = SAOutput(
                                logits=test_logits, 
                                hazard=test_hazard, 
                                survival=test_surv
                            )
                            
                            # Reinitialize test_refs with the small batch size
                            test_refs = torch.zeros(small_batch, 4 * num_events, requires_grad=False)
                            
                            # Try to compute loss
                            loss = loss_fn(test_preds, test_refs)
                            print(f"DEBUG: Loss computation successful: {loss.item()}")
                        except Exception as e:
                            print(f"DEBUG ERROR in {name}: {e}")
                    
                    # Run benchmark
                    forward_time, backward_time = run_loss_benchmark(
                        loss_fn, fresh_predictions, targets, num_iterations, name
                    )
                    
                    # Store results
                    results[f"{name}_forward"].append(forward_time)
                    results[f"{name}_backward"].append(backward_time)
                    results[f"{name}_total"].append(forward_time + backward_time)
                
                # Print a summary for this configuration
                print("Performance results:")
                for name in loss_names:
                    forward = results[f"{name}_forward"][-1]
                    backward = results[f"{name}_backward"][-1]
                    total = results[f"{name}_total"][-1]
                    print(f"  {name:20} - Forward: {forward:.4f} ms, Backward: {backward:.4f} ms, Total: {total:.4f} ms")
                
                # Calculate speedups relative to SampleRankingLoss for this configuration
                base_time = results["sample_ranking_total"][-1]
                if base_time > 0:
                    for name in loss_names:
                        if name != "sample_ranking":
                            speedup = base_time / results[f"{name}_total"][-1] if results[f"{name}_total"][-1] > 0 else 0
                            print(f"  {name:20} - Speedup: {speedup:.2f}x")
                
                print("")
                
            finally:
                # Clean up temporary files
                os.unlink(duration_cuts_file)
                os.unlink(importance_weights_file)
    
    return results


def save_results_to_csv(results, filename):
    """Save benchmark results to CSV file."""
    # Create logs directory if it doesn't exist
    log_dir = os.path.join(os.getcwd(), "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Prepare data for CSV
    data = []
    loss_names = [
        "sample_ranking", 
        "multi_event", 
        "sample_list_mle", 
        "survrnc",
        "sample_soap_uniform",
        "sample_soap_importance",
    ]
    
    # Extract unique configurations
    num_configs = len(results["batch_size"])
    
    for i in range(num_configs):
        batch_size = results["batch_size"][i]
        num_events = results["num_events"][i]
        
        # Get baseline time for calculating speedups
        base_time = results["sample_ranking_total"][i]
        
        # Add entry for each loss function
        for name in loss_names:
            forward_time = results[f"{name}_forward"][i]
            backward_time = results[f"{name}_backward"][i]
            total_time = results[f"{name}_total"][i]
            
            # Calculate speedup relative to baseline
            speedup = base_time / total_time if total_time > 0 else 0
            
            data.append({
                "batch_size": batch_size,
                "num_events": num_events,
                "method": name,
                "forward_time_ms": forward_time,
                "backward_time_ms": backward_time,
                "total_time_ms": total_time,
                "speedup": speedup if name != "sample_ranking" else 1.0
            })
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(data)
    csv_path = os.path.join(log_dir, filename)
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
    
    return csv_path


def generate_summary_markdown(csv_path):
    """Generate a summary markdown file with key findings from the benchmark results."""
    # Read CSV file
    df = pd.read_csv(csv_path)
    
    # Get unique methods excluding the baseline
    methods = [m for m in df["method"].unique() if m != "sample_ranking"]
    
    # Calculate average speedups by batch size for each method
    speedup_by_size = {}
    for method in methods:
        method_df = df[df["method"] == method]
        speedup_by_size[method] = method_df.groupby("batch_size")["speedup"].mean()
    
    # Calculate overall average speedups
    avg_speedups = {method: df[df["method"] == method]["speedup"].mean() for method in methods}
    max_speedups = {method: df[df["method"] == method]["speedup"].max() for method in methods}
    
    # Find the best method for each batch size
    best_by_size = {}
    for batch_size in df["batch_size"].unique():
        batch_df = df[df["batch_size"] == batch_size]
        methods_in_batch = [m for m in batch_df["method"].unique() if m != "sample_ranking"]
        best_method = max(methods_in_batch, key=lambda m: batch_df[batch_df["method"] == m]["speedup"].mean())
        best_speedup = batch_df[batch_df["method"] == best_method]["speedup"].mean()
        best_by_size[batch_size] = (best_method, best_speedup)
    
    # Generate markdown content
    markdown = "# Ranking Loss Performance Comparison\n\n"
    markdown += "This document summarizes the performance of different ranking loss implementations for survival analysis.\n\n"
    
    # Create comparison table by batch size
    markdown += "## Speedup by Batch Size\n\n"
    markdown += "The table below shows the average speedup factor compared to SampleRankingLoss (higher is better).\n\n"
    
    # Create header with method names
    markdown += "| Batch Size | " + " | ".join(methods) + " | Best Method |\n"
    markdown += "|" + "-" * 11 + "|" + "".join(["-" * (len(m) + 2) + "|" for m in methods]) + "-" * 14 + "|\n"
    
    # Add rows for each batch size
    for batch_size in sorted(df["batch_size"].unique()):
        row = f"| {batch_size} |"
        for method in methods:
            if batch_size in speedup_by_size[method]:
                row += f" {speedup_by_size[method][batch_size]:.2f}x |"
            else:
                row += " N/A |"
        
        best_method, best_speedup = best_by_size.get(batch_size, ("N/A", 0))
        if best_speedup > 0:
            row += f" {best_method} ({best_speedup:.2f}x) |"
        else:
            row += " N/A |"
        
        markdown += row + "\n"
    
    # Key findings
    markdown += "\n## Key Findings\n\n"
    for method in methods:
        markdown += f"- **{method}**: Average {avg_speedups[method]:.2f}x speedup, Max {max_speedups[method]:.2f}x\n"
    
    # Best overall method
    best_method = max(methods, key=lambda m: avg_speedups[m])
    markdown += f"\n**Overall Best Performer**: {best_method} (Average: {avg_speedups[best_method]:.2f}x)\n\n"
    
    # Characteristics of different methods
    markdown += "\n## Method Characteristics\n\n"
    
    markdown += "### SampleRankingLoss (Baseline)\n"
    markdown += "- Traditional pairwise ranking approach\n"
    markdown += "- O(n²) complexity with batch size\n"
    markdown += "- Complete pairwise comparison of all samples\n\n"
    
    markdown += "### MultiEventRankingLoss\n"
    markdown += "- Compares different event types within the same observation\n"
    markdown += "- O(e²) complexity with number of events\n"
    markdown += "- Complementary to SampleRankingLoss\n\n"
    
    markdown += "### SampleListMLELoss\n"
    markdown += "- Listwise ranking approach using Plackett-Luce model\n"
    markdown += "- Better scaling than pairwise: O(n log n)\n"
    markdown += "- Directly optimizes the probability of correct ordering\n\n"
    
    markdown += "### SurvRNCLoss\n"
    markdown += "- Combines ranking with contrastive learning\n"
    markdown += "- O(n²) complexity but with hard mining option for large batches\n"
    markdown += "- Focuses on separating samples with different outcomes\n\n"
    
    markdown += "### SampleSOAPLoss (Uniform)\n"
    markdown += "- Accelerated pairwise approach with uniform pair sampling\n"
    markdown += "- O(n log n) complexity through strategic pair selection\n"
    markdown += "- Maintains statistical optimality with fewer comparisons\n\n"
    
    markdown += "### SampleSOAPLoss (Importance)\n"
    markdown += "- Importance-weighted pair sampling strategy\n"
    markdown += "- Focuses on pairs with larger duration differences\n"
    markdown += "- Adaptive margin based on duration differences\n\n"
    
    # Recommendations
    markdown += "## Recommendations\n\n"
    markdown += "Based on the benchmark results, the following recommendations can be made:\n\n"
    
    # Small batch sizes
    small_batch = min(df["batch_size"].unique())
    small_batch_best, small_batch_speedup = best_by_size.get(small_batch, ("N/A", 0))
    markdown += f"- **Small Batches (≤ {small_batch})**: {small_batch_best} "
    markdown += f"offers the best performance with {small_batch_speedup:.2f}x speedup.\n\n"
    
    # Large batch sizes
    large_batch = max(df["batch_size"].unique())
    large_batch_best, large_batch_speedup = best_by_size.get(large_batch, ("N/A", 0))
    markdown += f"- **Large Batches (≥ {large_batch})**: {large_batch_best} "
    markdown += f"offers the best performance with {large_batch_speedup:.2f}x speedup.\n\n"
    
    # General recommendation
    markdown += f"- **General Use**: {best_method} provides the best overall performance "
    markdown += f"across different batch sizes and event counts.\n\n"
    
    # Save markdown to file
    log_dir = os.path.join(os.getcwd(), "logs")
    md_path = os.path.join(log_dir, "ranking_loss_comparison.md")
    
    with open(md_path, "w") as f:
        f.write(markdown)
    
    print(f"Summary saved to {md_path}")
    return md_path


def plot_benchmark_results(results, save_path=None):
    """Plot benchmark results."""
    # Extract unique batch sizes and event counts
    batch_sizes = sorted(set(results["batch_size"]))
    num_events_set = sorted(set(results["num_events"]))
    
    # Define methods and their display names
    methods = [
        "sample_ranking", 
        "multi_event", 
        "sample_list_mle", 
        "survrnc",
        "sample_soap_uniform",
        "sample_soap_importance",
    ]
    
    method_names = {
        "sample_ranking": "SampleRanking",
        "multi_event": "MultiEvent",
        "sample_list_mle": "SampleListMLE",
        "survrnc": "SurvRNC",
        "sample_soap_uniform": "SOAP-Uniform",
        "sample_soap_importance": "SOAP-Importance",
    }
    
    # Colors for plotting
    colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
    
    # Create subplots for each num_events value
    fig, axes = plt.subplots(len(num_events_set), 2, figsize=(16, 6 * len(num_events_set)))
    if len(num_events_set) == 1:
        axes = np.array([axes])  # Ensure axes is 2D for consistent indexing
    
    for i, num_events in enumerate(num_events_set):
        # Filter results for this number of events
        indices = [j for j, ne in enumerate(results["num_events"]) if ne == num_events]
        
        # Get times and batch sizes for this event count
        batch_sizes_filtered = [results["batch_size"][j] for j in indices]
        
        # Execution time plot (log scale)
        ax1 = axes[i, 0]
        for c, method in enumerate(methods):
            times = [results[f"{method}_total"][j] for j in indices]
            ax1.plot(batch_sizes_filtered, times, 'o-', label=method_names[method], color=colors[c])
        
        ax1.set_title(f"Execution Time (Events={num_events})")
        ax1.set_xlabel("Batch Size")
        ax1.set_ylabel("Time (ms)")
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)
        if i == 0:  # Only add legend to first row
            ax1.legend()
        
        # Speedup plot
        ax2 = axes[i, 1]
        for c, method in enumerate(methods):
            if method == "sample_ranking":
                continue  # Skip baseline
            
            # Get baseline times
            baseline_times = [results["sample_ranking_total"][j] for j in indices]
            # Get method times
            method_times = [results[f"{method}_total"][j] for j in indices]
            # Calculate speedups
            speedups = [b / m if m > 0 else 0 for b, m in zip(baseline_times, method_times)]
            
            ax2.plot(batch_sizes_filtered, speedups, 'o-', label=method_names[method], color=colors[c])
        
        ax2.set_title(f"Speedup vs. SampleRanking (Events={num_events})")
        ax2.set_xlabel("Batch Size")
        ax2.set_ylabel("Speedup Factor")
        ax2.axhline(y=1.0, color='k', linestyle='--', alpha=0.3)
        ax2.grid(True, alpha=0.3)
        if i == 0:  # Only add legend to first row
            ax2.legend()
    
    plt.tight_layout()
    
    # Save plot if path provided
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    
    # Show plot
    plt.show()


def main():
    """Run benchmarks and generate reports."""
    print("Running benchmarks for all ranking losses...")
    results = benchmark_all_losses(
        batch_sizes=[16, 32, 64, 128, 256], 
        num_events_list=[1, 2], 
        num_cuts=10, 
        num_iterations=5
    )
    
    # Save results to CSV
    csv_path = save_results_to_csv(results, "ranking_loss_benchmark.csv")
    
    # Generate summary markdown
    summary_md = generate_summary_markdown(csv_path)
    
    # Plot results if matplotlib is available
    try:
        plot_path = os.path.join(os.getcwd(), "logs", "ranking_loss_comparison.png")
        plot_benchmark_results(results, save_path=plot_path)
    except Exception as e:
        print(f"Error creating plots: {e}")


if __name__ == "__main__":
    main()