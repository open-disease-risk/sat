"""Benchmark script for MENSA model versus other survival models"""

import os
import sys
import time
import tempfile
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import argparse

from sat.models.heads.dsm import DSMConfig, DSMTaskHead
from sat.models.heads.mensa import MENSAConfig, MENSATaskHead
from sat.loss.survival.dsm import DSMLoss
from sat.loss.survival.mensa import MENSALoss
from sat.models.heads import SAOutput


def create_duration_cuts_file(num_cuts: int = 10, max_duration: float = 600) -> str:
    """Create a temporary file with duration cuts."""
    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        duration_cuts = np.linspace(1, max_duration, num_cuts)
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


def generate_synthetic_data(
    num_samples: int, num_features: int, num_events: int, seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate synthetic data for benchmarking.

    Args:
        num_samples: Number of samples to generate
        num_features: Number of features per sample
        num_events: Number of competing events
        seed: Random seed for reproducibility

    Returns:
        Tuple of (features, labels)
    """
    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Generate features
    features = torch.randn(num_samples, num_features)

    # Generate reference values
    # Format: [percentiles, event indicators, fractions, durations] * num_events
    references = torch.zeros(num_samples, 4 * num_events)

    # Set event indicators - distribute samples evenly among events and censoring
    event_prob = 0.7  # Probability of any event occurring
    samples_per_event = int(num_samples * event_prob / num_events)

    for event_idx in range(num_events):
        start_idx = event_idx * samples_per_event
        end_idx = (event_idx + 1) * samples_per_event
        references[start_idx:end_idx, num_events + event_idx] = 1

    # Generate durations based on features (create correlation)
    for i in range(num_samples):
        for event_idx in range(num_events):
            # Generate duration based on selected features
            feature_sum = features[i, :5].sum().item()  # Use first 5 features

            # Normalize to reasonable duration range (1-10)
            duration = abs(feature_sum) * 2 + 1  # Will range roughly from 1 to 10

            # Add random noise
            duration += np.random.normal(0, 0.5)

            # Ensure positive
            duration = max(0.1, duration)

            # Store in references tensor
            references[i, 3 * num_events + event_idx] = duration

    return features, references


def setup_models(
    num_features: int,
    num_events: int,
    num_mixtures: int,
    temp_files: Dict[str, str],
    device: torch.device,
) -> Dict[str, torch.nn.Module]:
    """
    Set up models for benchmarking.

    Args:
        num_features: Number of input features
        num_events: Number of competing events
        num_mixtures: Number of mixture components
        temp_files: Dictionary of temporary file paths
        device: PyTorch device to run models on

    Returns:
        Dictionary of model names to initialized models
    """
    # Create DSM model
    dsm_config = DSMConfig(
        num_features=num_features,
        intermediate_size=64,
        num_hidden_layers=2,
        indiv_intermediate_size=32,
        indiv_num_hidden_layers=1,
        num_mixtures=num_mixtures,
        num_events=num_events,
        distribution="weibull",
    )

    dsm_model = DSMTaskHead(dsm_config).to(device)

    # Create DSM loss
    dsm_loss = DSMLoss(
        duration_cuts=temp_files["duration_cuts"],
        importance_sample_weights=temp_files["importance_weights"],
        num_events=num_events,
        distribution="weibull",
    )

    dsm_model.loss = dsm_loss

    # Create MENSA model without event dependencies
    mensa_config_no_dep = MENSAConfig(
        num_features=num_features,
        intermediate_size=64,
        num_hidden_layers=2,
        indiv_intermediate_size=32,
        indiv_num_hidden_layers=1,
        num_mixtures=num_mixtures,
        num_events=num_events,
        event_dependency=False,
        distribution="weibull",
    )

    mensa_model_no_dep = MENSATaskHead(mensa_config_no_dep).to(device)

    # Create MENSA loss
    mensa_loss_no_dep = MENSALoss(
        duration_cuts=temp_files["duration_cuts"],
        importance_sample_weights=temp_files["importance_weights"],
        num_events=num_events,
        distribution="weibull",
        dependency_regularization=0.0,  # No regularization for no dependency model
    )

    mensa_model_no_dep.loss = mensa_loss_no_dep

    # Create MENSA model with event dependencies
    mensa_config_with_dep = MENSAConfig(
        num_features=num_features,
        intermediate_size=64,
        num_hidden_layers=2,
        indiv_intermediate_size=32,
        indiv_num_hidden_layers=1,
        num_mixtures=num_mixtures,
        num_events=num_events,
        event_dependency=True,
        distribution="weibull",
    )

    mensa_model_with_dep = MENSATaskHead(mensa_config_with_dep).to(device)

    # Create MENSA loss with dependency regularization
    mensa_loss_with_dep = MENSALoss(
        duration_cuts=temp_files["duration_cuts"],
        importance_sample_weights=temp_files["importance_weights"],
        num_events=num_events,
        distribution="weibull",
        dependency_regularization=0.01,  # Add regularization
    )

    mensa_model_with_dep.loss = mensa_loss_with_dep

    return {
        "DSM": dsm_model,
        "MENSA (no dependencies)": mensa_model_no_dep,
        "MENSA (with dependencies)": mensa_model_with_dep,
    }


def train_epoch(
    model: torch.nn.Module,
    features: torch.Tensor,
    references: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    device: torch.device,
) -> float:
    """
    Train model for one epoch.

    Args:
        model: Model to train
        features: Input features
        references: Reference labels
        optimizer: Optimizer
        batch_size: Batch size
        device: PyTorch device

    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    # Create indices
    indices = torch.randperm(features.size(0))

    # Process in batches
    for start_idx in range(0, features.size(0), batch_size):
        # Get batch indices
        batch_indices = indices[start_idx : start_idx + batch_size]

        # Get batch data
        batch_features = features[batch_indices].to(device)
        batch_references = references[batch_indices].to(device)

        # Forward pass
        optimizer.zero_grad()
        output = model(batch_features, batch_references)

        # Backward pass
        output.loss.backward()
        optimizer.step()

        # Track loss
        total_loss += output.loss.item()
        num_batches += 1

    return total_loss / num_batches


def evaluate_model(
    model: torch.nn.Module,
    features: torch.Tensor,
    references: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluate model on test data.

    Args:
        model: Model to evaluate
        features: Input features
        references: Reference labels
        batch_size: Batch size
        device: PyTorch device

    Returns:
        Dictionary of metrics
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    # Create arrays to store predictions
    all_pred_risks = []
    all_event_indicators = []
    all_durations = []

    # Process in batches
    with torch.no_grad():
        for start_idx in range(0, features.size(0), batch_size):
            # Get batch data
            end_idx = min(start_idx + batch_size, features.size(0))
            batch_features = features[start_idx:end_idx].to(device)
            batch_references = references[start_idx:end_idx].to(device)

            # Forward pass
            output = model(batch_features, batch_references)

            # Track loss
            total_loss += output.loss.item()
            num_batches += 1

            # Store predictions for metric calculation
            # Extract risk at last time point for each event
            final_risks = output.risk[:, :, -1].cpu()
            all_pred_risks.append(final_risks)

            # Extract event indicators
            num_events = output.risk.size(1)
            event_indicators = torch.zeros_like(
                batch_references[:, num_events : 2 * num_events]
            )
            for i in range(num_events):
                event_indicators[:, i] = batch_references[:, num_events + i]
            all_event_indicators.append(event_indicators.cpu())

            # Extract durations
            durations = batch_references[:, 3 * num_events : 4 * num_events].cpu()
            all_durations.append(durations)

    # Concatenate predictions from all batches
    all_pred_risks = torch.cat(all_pred_risks, dim=0)
    all_event_indicators = torch.cat(all_event_indicators, dim=0)
    all_durations = torch.cat(all_durations, dim=0)

    # Compute C-index (simplified version for benchmarking)
    c_index = compute_c_index(all_pred_risks, all_event_indicators, all_durations)

    # Return metrics
    return {
        "loss": total_loss / num_batches,
        "c_index": c_index,
    }


def compute_c_index(pred_risks, event_indicators, durations):
    """
    Compute simplified C-index for competing risks (one c-index per event type).
    """
    num_events = pred_risks.size(1)
    c_indices = []

    for event_idx in range(num_events):
        # Get predictions, indicators, and durations for this event
        risks = pred_risks[:, event_idx]
        events = event_indicators[:, event_idx]
        times = durations[:, event_idx]

        # Find samples with this event
        event_mask = events == 1
        if not torch.any(event_mask):
            # No samples with this event
            c_indices.append(0.0)
            continue

        # Compute concordant pairs
        concordant = 0
        comparable = 0

        # Get event samples
        event_indices = torch.where(event_mask)[0]

        for i in event_indices:
            # Compare with all other samples
            for j in range(len(risks)):
                if i == j:
                    continue

                # Samples are comparable if i had an event and j's time > i's time
                if times[j] > times[i]:
                    comparable += 1

                    # Concordant if higher risk prediction for the event case
                    if risks[i] > risks[j]:
                        concordant += 1

        # Compute c-index for this event
        if comparable > 0:
            c_indices.append(concordant / comparable)
        else:
            c_indices.append(0.0)

    # Return average c-index across events
    return sum(c_indices) / len(c_indices)


def benchmark_models(
    models: Dict[str, torch.nn.Module],
    train_features: torch.Tensor,
    train_references: torch.Tensor,
    test_features: torch.Tensor,
    test_references: torch.Tensor,
    num_epochs: int,
    batch_size: int,
    device: torch.device,
) -> Dict[str, Dict[str, List[float]]]:
    """
    Benchmark models on training and evaluation.

    Args:
        models: Dictionary of models to benchmark
        train_features: Training features
        train_references: Training references
        test_features: Test features
        test_references: Test references
        num_epochs: Number of epochs to train
        batch_size: Batch size
        device: PyTorch device

    Returns:
        Dictionary of metrics per model
    """
    results = {}

    for model_name, model in models.items():
        print(f"Benchmarking {model_name}...")

        # Initialize optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Track metrics
        train_losses = []
        eval_losses = []
        c_indices = []
        train_times = []
        inference_times = []

        # Train and evaluate
        for epoch in range(num_epochs):
            # Time training
            start_time = time.time()
            train_loss = train_epoch(
                model, train_features, train_references, optimizer, batch_size, device
            )
            train_time = time.time() - start_time
            train_times.append(train_time)
            train_losses.append(train_loss)

            # Time evaluation
            start_time = time.time()
            metrics = evaluate_model(
                model, test_features, test_references, batch_size, device
            )
            inference_time = time.time() - start_time
            inference_times.append(inference_time)

            # Track metrics
            eval_losses.append(metrics["loss"])
            c_indices.append(metrics["c_index"])

            # Print progress
            print(
                f"  Epoch {epoch+1}/{num_epochs}: "
                f"Train Loss: {train_loss:.4f}, "
                f"Eval Loss: {metrics['loss']:.4f}, "
                f"C-index: {metrics['c_index']:.4f}, "
                f"Train Time: {train_time:.4f}s, "
                f"Inference Time: {inference_time:.4f}s"
            )

        # Store results
        results[model_name] = {
            "train_loss": train_losses,
            "eval_loss": eval_losses,
            "c_index": c_indices,
            "train_time": train_times,
            "inference_time": inference_times,
        }

        # Check dependency matrix for MENSA models with dependencies
        if "with dependencies" in model_name and hasattr(
            model.nets, "event_dependency_matrix"
        ):
            dependency_matrix = torch.softmax(model.nets.event_dependency_matrix, dim=1)
            print(
                f"  Learned dependency matrix:\n{dependency_matrix.detach().cpu().numpy()}"
            )

    return results


def plot_results(results: Dict[str, Dict[str, List[float]]], output_dir: str):
    """
    Plot benchmark results.

    Args:
        results: Dictionary of metrics per model
        output_dir: Directory to save plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Plot training loss
    plt.figure(figsize=(10, 6))
    for model_name, metrics in results.items():
        plt.plot(metrics["train_loss"], label=model_name)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Training Loss per Epoch")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.savefig(os.path.join(output_dir, "train_loss.png"))

    # Plot evaluation loss
    plt.figure(figsize=(10, 6))
    for model_name, metrics in results.items():
        plt.plot(metrics["eval_loss"], label=model_name)
    plt.xlabel("Epoch")
    plt.ylabel("Evaluation Loss")
    plt.title("Evaluation Loss per Epoch")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.savefig(os.path.join(output_dir, "eval_loss.png"))

    # Plot C-index
    plt.figure(figsize=(10, 6))
    for model_name, metrics in results.items():
        plt.plot(metrics["c_index"], label=model_name)
    plt.xlabel("Epoch")
    plt.ylabel("C-index")
    plt.title("C-index per Epoch")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.savefig(os.path.join(output_dir, "c_index.png"))

    # Plot training time
    plt.figure(figsize=(10, 6))
    avg_train_times = [np.mean(metrics["train_time"]) for metrics in results.values()]
    plt.bar(results.keys(), avg_train_times)
    plt.xlabel("Model")
    plt.ylabel("Average Training Time (s)")
    plt.title("Average Training Time per Epoch")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "train_time.png"))

    # Plot inference time
    plt.figure(figsize=(10, 6))
    avg_inference_times = [
        np.mean(metrics["inference_time"]) for metrics in results.values()
    ]
    plt.bar(results.keys(), avg_inference_times)
    plt.xlabel("Model")
    plt.ylabel("Average Inference Time (s)")
    plt.title("Average Inference Time per Epoch")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "inference_time.png"))

    # Save results as CSV
    results_df = pd.DataFrame(
        {
            "Model": [],
            "Metric": [],
            "Value": [],
        }
    )

    for model_name, metrics in results.items():
        # Extract final values
        final_train_loss = metrics["train_loss"][-1]
        final_eval_loss = metrics["eval_loss"][-1]
        final_c_index = metrics["c_index"][-1]
        avg_train_time = np.mean(metrics["train_time"])
        avg_inference_time = np.mean(metrics["inference_time"])

        # Add to dataframe
        results_df = pd.concat(
            [
                results_df,
                pd.DataFrame(
                    {
                        "Model": [model_name] * 5,
                        "Metric": [
                            "Train Loss",
                            "Eval Loss",
                            "C-index",
                            "Avg Train Time (s)",
                            "Avg Inference Time (s)",
                        ],
                        "Value": [
                            final_train_loss,
                            final_eval_loss,
                            final_c_index,
                            avg_train_time,
                            avg_inference_time,
                        ],
                    }
                ),
            ],
            ignore_index=True,
        )

    # Save to CSV
    results_df.to_csv(os.path.join(output_dir, "benchmark_results.csv"), index=False)

    # Print summary
    print("\nBenchmark Summary:")
    print(results_df.to_string(index=False))


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Benchmark MENSA vs other survival models"
    )
    parser.add_argument(
        "--num_samples", type=int, default=1000, help="Number of samples"
    )
    parser.add_argument(
        "--num_features", type=int, default=32, help="Number of features"
    )
    parser.add_argument(
        "--num_events", type=int, default=2, help="Number of competing events"
    )
    parser.add_argument(
        "--num_mixtures", type=int, default=4, help="Number of mixture components"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--output_dir", type=str, default="benchmark_outputs", help="Output directory"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA")
    args = parser.parse_args()

    # Set device
    device = torch.device("cpu")
    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and not args.no_cuda:
        device = torch.device("mps")  # For Apple Silicon
    print(f"Using device: {device}")

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create temporary files
    temp_files = {
        "duration_cuts": create_duration_cuts_file(num_cuts=20),
        "importance_weights": create_importance_weights_file(
            num_events=args.num_events
        ),
    }

    try:
        # Generate synthetic data
        print(
            f"Generating synthetic data with {args.num_samples} samples, "
            f"{args.num_features} features, {args.num_events} events..."
        )
        features, references = generate_synthetic_data(
            args.num_samples, args.num_features, args.num_events, args.seed
        )

        # Split into train/test
        train_size = int(0.8 * args.num_samples)
        train_features, test_features = features[:train_size], features[train_size:]
        train_references, test_references = (
            references[:train_size],
            references[train_size:],
        )

        # Set up models
        print("Setting up models...")
        models = setup_models(
            args.num_features, args.num_events, args.num_mixtures, temp_files, device
        )

        # Run benchmark
        print(f"Running benchmark for {args.num_epochs} epochs...")
        results = benchmark_models(
            models,
            train_features,
            train_references,
            test_features,
            test_references,
            args.num_epochs,
            args.batch_size,
            device,
        )

        # Plot results
        print(f"Plotting results to {args.output_dir}...")
        plot_results(results, args.output_dir)

    finally:
        # Clean up temporary files
        for file_path in temp_files.values():
            try:
                os.unlink(file_path)
            except:
                pass


if __name__ == "__main__":
    main()
