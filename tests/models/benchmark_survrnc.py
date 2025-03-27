"""Benchmark SurvRNCLoss against other ranking losses for survival analysis."""

import os
import tempfile
import pandas as pd
import torch
import time
import numpy as np
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt

from sat.models.heads import SAOutput
from sat.loss.ranking.multievent import MultiEventRankingLoss
from sat.loss.ranking.sample import SampleRankingLoss
from sat.loss.ranking.sample_list_mle import SampleListMLELoss
from sat.loss.ranking.survrnc import SurvRNCLoss
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


def benchmark_all_losses(batch_sizes=[16, 32], num_events_list=[1, 2], num_cuts=10, num_iterations=3):
    """Benchmark all ranking losses for different batch sizes and event counts."""
    # Results dictionary
    results = {
        "batch_size": [],
        "num_events": [],
        "multievent_forward": [],
        "sample_forward": [],
        "sample_list_mle_forward": [],
        "survrnc_forward": [],
        "multievent_backward": [],
        "sample_backward": [],
        "sample_list_mle_backward": [],
        "survrnc_backward": [],
    }
    
    # Run benchmarks for each configuration
    for batch_size in batch_sizes:
        for num_events in num_events_list:
            print(f"Benchmarking batch_size={batch_size}, num_events={num_events}")
            
            # Create files needed for loss initialization
            duration_cuts_file = create_duration_cuts_file(num_cuts)
            importance_weights_file = create_importance_weights_file(num_events)
            
            try:
                # Create loss instances
                multievent_loss = MultiEventRankingLoss(
                    duration_cuts=duration_cuts_file,
                    importance_sample_weights=importance_weights_file,
                    num_events=num_events,
                    sigma=0.1,
                    margin=0.05,
                )
                
                sample_loss = SampleRankingLoss(
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
                )
                
                # Create test data
                # Create new data for each iteration
                hazard = torch.rand(
                    batch_size, num_events, num_cuts, requires_grad=True
                )
                ones = torch.ones(batch_size, num_events, 1)
                survival_base = (
                    1
                    - torch.cumsum(torch.nn.functional.softplus(hazard), dim=2)
                    / num_cuts
                )
                survival = torch.cat([ones, survival_base], dim=2)
                logits = torch.zeros_like(hazard)
                predictions = SAOutput(logits=logits, hazard=hazard, survival=survival)

                _, targets = create_fake_data(batch_size, num_events, num_cuts)
                
                # Run benchmarks
                multi_forward, multi_backward = run_loss_benchmark(
                    multievent_loss, predictions, targets, num_iterations, "MultiEventRankingLoss"
                )
                
                # Reset gradients
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Create new data
                hazard = torch.rand(
                    batch_size, num_events, num_cuts, requires_grad=True
                )
                ones = torch.ones(batch_size, num_events, 1)
                survival_base = (
                    1 - torch.cumsum(torch.nn.functional.softplus(hazard), dim=2) / num_cuts
                )
                survival = torch.cat([ones, survival_base], dim=2)
                logits = torch.zeros_like(hazard)
                predictions = SAOutput(logits=logits, hazard=hazard, survival=survival)
                
                sample_forward, sample_backward = run_loss_benchmark(
                    sample_loss, predictions, targets, num_iterations, "SampleRankingLoss"
                )
                
                # Reset gradients
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Create new data
                hazard = torch.rand(
                    batch_size, num_events, num_cuts, requires_grad=True
                )
                ones = torch.ones(batch_size, num_events, 1)
                survival_base = (
                    1 - torch.cumsum(torch.nn.functional.softplus(hazard), dim=2) / num_cuts
                )
                survival = torch.cat([ones, survival_base], dim=2)
                logits = torch.zeros_like(hazard)
                predictions = SAOutput(logits=logits, hazard=hazard, survival=survival)
                
                sample_list_mle_forward, sample_list_mle_backward = run_loss_benchmark(
                    sample_list_mle_loss, predictions, targets, num_iterations, "SampleListMLELoss"
                )
                
                # Reset gradients
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Create new data
                hazard = torch.rand(
                    batch_size, num_events, num_cuts, requires_grad=True
                )
                ones = torch.ones(batch_size, num_events, 1)
                survival_base = (
                    1 - torch.cumsum(torch.nn.functional.softplus(hazard), dim=2) / num_cuts
                )
                survival = torch.cat([ones, survival_base], dim=2)
                logits = torch.zeros_like(hazard)
                predictions = SAOutput(logits=logits, hazard=hazard, survival=survival)
                
                survrnc_forward, survrnc_backward = run_loss_benchmark(
                    survrnc_loss, predictions, targets, num_iterations, "SurvRNCLoss"
                )
                
                # Store results
                results["batch_size"].append(batch_size)
                results["num_events"].append(num_events)
                results["multievent_forward"].append(multi_forward)
                results["sample_forward"].append(sample_forward)
                results["sample_list_mle_forward"].append(sample_list_mle_forward)
                results["survrnc_forward"].append(survrnc_forward)
                results["multievent_backward"].append(multi_backward)
                results["sample_backward"].append(sample_backward)
                results["sample_list_mle_backward"].append(sample_list_mle_backward)
                results["survrnc_backward"].append(survrnc_backward)
                
                # Print results
                print(f"  MultiEventRankingLoss   - Forward: {multi_forward:.4f} ms, Backward: {multi_backward:.4f} ms")
                print(f"  SampleRankingLoss       - Forward: {sample_forward:.4f} ms, Backward: {sample_backward:.4f} ms")
                print(f"  SampleListMLELoss       - Forward: {sample_list_mle_forward:.4f} ms, Backward: {sample_list_mle_backward:.4f} ms")
                print(f"  SurvRNCLoss             - Forward: {survrnc_forward:.4f} ms, Backward: {survrnc_backward:.4f} ms")
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
    
    # Extract unique batch sizes and event counts
    batch_sizes = set(results["batch_size"])
    num_events_list = set(results["num_events"])
    
    for i in range(len(results["batch_size"])):
        batch_size = results["batch_size"][i]
        num_events = results["num_events"][i]
        
        # MultiEventRankingLoss
        data.append({
            "batch_size": batch_size,
            "num_events": num_events,
            "method": "MultiEventRankingLoss",
            "forward_time_ms": results["multievent_forward"][i],
            "backward_time_ms": results["multievent_backward"][i],
            "total_time_ms": results["multievent_forward"][i] + results["multievent_backward"][i]
        })
        
        # SampleRankingLoss
        data.append({
            "batch_size": batch_size,
            "num_events": num_events,
            "method": "SampleRankingLoss",
            "forward_time_ms": results["sample_forward"][i],
            "backward_time_ms": results["sample_backward"][i],
            "total_time_ms": results["sample_forward"][i] + results["sample_backward"][i]
        })
        
        # SampleListMLELoss
        data.append({
            "batch_size": batch_size,
            "num_events": num_events,
            "method": "SampleListMLELoss",
            "forward_time_ms": results["sample_list_mle_forward"][i],
            "backward_time_ms": results["sample_list_mle_backward"][i],
            "total_time_ms": results["sample_list_mle_forward"][i] + results["sample_list_mle_backward"][i]
        })
        
        # SurvRNCLoss
        data.append({
            "batch_size": batch_size,
            "num_events": num_events,
            "method": "SurvRNCLoss",
            "forward_time_ms": results["survrnc_forward"][i],
            "backward_time_ms": results["survrnc_backward"][i],
            "total_time_ms": results["survrnc_forward"][i] + results["survrnc_backward"][i]
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
    
    # Calculate speedups compared to SampleRankingLoss
    summary = {}
    
    # Get unique configurations
    configs = df[['batch_size', 'num_events']].drop_duplicates().values
    
    for batch_size, num_events in configs:
        # Filter data for this configuration
        config_df = df[(df['batch_size'] == batch_size) & (df['num_events'] == num_events)]
        
        # Get reference time (SampleRankingLoss)
        reference = config_df[config_df['method'] == 'SampleRankingLoss']['total_time_ms'].values[0]
        
        # Calculate speedups for each method
        config_key = f"b{batch_size}_e{num_events}"
        summary[config_key] = {}
        
        for method in config_df['method'].unique():
            if method == 'SampleRankingLoss':
                continue
                
            method_time = config_df[config_df['method'] == method]['total_time_ms'].values[0]
            speedup = reference / method_time if method_time > 0 else 0
            summary[config_key][method] = speedup
    
    # Generate markdown content
    markdown = "# SurvRNC vs Other Ranking Losses - Benchmark Summary\n\n"
    markdown += "This document summarizes the performance comparison between different ranking loss implementations, including the new SurvRNCLoss.\n\n"
    
    # Create comparison table
    markdown += "## Performance Comparison\n\n"
    markdown += "The table below shows the speedup factor compared to SampleRankingLoss (higher is better).\n\n"
    
    markdown += "| Configuration | MultiEventRankingLoss | SampleListMLELoss | SurvRNCLoss |\n"
    markdown += "|---------------|----------------------|-------------------|------------|\n"
    
    for config_key, values in summary.items():
        markdown += f"| {config_key} | {values.get('MultiEventRankingLoss', 'N/A'):.2f}x | {values.get('SampleListMLELoss', 'N/A'):.2f}x | {values.get('SurvRNCLoss', 'N/A'):.2f}x |\n"
    
    # Overall findings
    markdown += "\n## Key Findings\n\n"
    
    # Calculate average speedups across all configurations
    all_multi = [values.get('MultiEventRankingLoss', 0) for values in summary.values()]
    all_sample_list = [values.get('SampleListMLELoss', 0) for values in summary.values()]
    all_survrnc = [values.get('SurvRNCLoss', 0) for values in summary.values()]
    
    avg_multi = np.mean(all_multi) if all_multi else 0
    avg_sample_list = np.mean(all_sample_list) if all_sample_list else 0
    avg_survrnc = np.mean(all_survrnc) if all_survrnc else 0
    
    markdown += f"- **MultiEventRankingLoss**: Average {avg_multi:.2f}x speedup compared to SampleRankingLoss\n"
    markdown += f"- **SampleListMLELoss**: Average {avg_sample_list:.2f}x speedup compared to SampleRankingLoss\n"
    markdown += f"- **SurvRNCLoss**: Average {avg_survrnc:.2f}x speedup compared to SampleRankingLoss\n\n"
    
    # Best performing method
    best_method = "MultiEventRankingLoss" if avg_multi > max(avg_sample_list, avg_survrnc) else (
        "SampleListMLELoss" if avg_sample_list > avg_survrnc else "SurvRNCLoss"
    )
    
    markdown += f"**Overall Best Performer**: {best_method}\n\n"
    
    # Add SurvRNC specific characteristics
    markdown += "\n## SurvRNC Loss Characteristics\n\n"
    markdown += "The Survival Rank-N-Contrast (SurvRNC) loss has the following characteristics:\n\n"
    markdown += "1. **Contrastive Learning Approach**: Unlike pairwise ranking losses, SurvRNC uses an N-pair contrastive approach that focuses on learning similarity between samples with similar outcomes.\n\n"
    markdown += "2. **Improved Generalization**: By focusing on grouping similar patients together in the embedding space, SurvRNC can lead to better generalization performance, especially in cases with limited data.\n\n"
    markdown += "3. **Temperature Parameter**: Allows for controlling the sharpness of the similarity distribution, which can be important for balancing between hard and soft contrasts.\n\n"
    markdown += "4. **Margin Parameter**: Controls the separation boundary between samples considered similar vs. dissimilar, which can help with robustness.\n\n"
    markdown += "5. **Computational Efficiency**: SurvRNC has a computational complexity that scales better with dataset size compared to traditional pairwise ranking methods.\n\n"
    
    # Save markdown to file
    log_dir = os.path.join(os.getcwd(), "logs")
    md_path = os.path.join(log_dir, "survrnc_comparison.md")
    
    with open(md_path, "w") as f:
        f.write(markdown)
    
    print(f"Summary saved to {md_path}")
    return md_path


def plot_benchmark_results(results, save_path=None):
    """Plot benchmark results."""
    # Extract unique configurations
    configs = []
    for i in range(len(results["batch_size"])):
        configs.append(f"b{results['batch_size'][i]}_e{results['num_events'][i]}")
    
    # Set up plot
    plt.figure(figsize=(14, 8))
    
    # Bar width
    width = 0.2
    
    # Set positions for bar groups
    positions = np.arange(len(configs))
    
    # Plot forward times
    plt.subplot(1, 2, 1)
    plt.bar(positions - 1.5*width, results["multievent_forward"], width, label='MultiEventRankingLoss')
    plt.bar(positions - 0.5*width, results["sample_forward"], width, label='SampleRankingLoss')
    plt.bar(positions + 0.5*width, results["sample_list_mle_forward"], width, label='SampleListMLELoss')
    plt.bar(positions + 1.5*width, results["survrnc_forward"], width, label='SurvRNCLoss')
    
    plt.xlabel('Configuration')
    plt.ylabel('Forward Time (ms)')
    plt.title('Forward Pass Performance')
    plt.xticks(positions, configs)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    
    # Plot backward times
    plt.subplot(1, 2, 2)
    plt.bar(positions - 1.5*width, results["multievent_backward"], width, label='MultiEventRankingLoss')
    plt.bar(positions - 0.5*width, results["sample_backward"], width, label='SampleRankingLoss')
    plt.bar(positions + 0.5*width, results["sample_list_mle_backward"], width, label='SampleListMLELoss')
    plt.bar(positions + 1.5*width, results["survrnc_backward"], width, label='SurvRNCLoss')
    
    plt.xlabel('Configuration')
    plt.ylabel('Backward Time (ms)')
    plt.title('Backward Pass Performance')
    plt.xticks(positions, configs)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path)
    
    # Show plot
    plt.show()


def main():
    """Run benchmarks and generate reports."""
    print("Running benchmarks...")
    results = benchmark_all_losses(
        batch_sizes=[16, 32], 
        num_events_list=[1, 2], 
        num_cuts=10, 
        num_iterations=3
    )
    
    # Save results to CSV
    csv_path = save_results_to_csv(results, "survrnc_benchmark.csv")
    
    # Generate summary markdown
    summary_md = generate_summary_markdown(csv_path)
    
    # Plot results if matplotlib is available
    try:
        plot_benchmark_results(results, save_path="logs/survrnc_performance.png")
    except Exception as e:
        print(f"Error creating plots: {e}")


if __name__ == "__main__":
    main()