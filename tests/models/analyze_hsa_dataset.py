"""Analyze HSA synthetic dataset with MultiEventRankingLoss."""

import os
import tempfile
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from sat.loss.ranking.multievent import MultiEventRankingLoss
from sat.loss.ranking.sample import SampleRankingLoss
from sat.models.heads import SAOutput


def load_hsa_synthetic_data(file_path: str) -> pd.DataFrame:
    """Load HSA synthetic dataset."""
    return pd.read_csv(file_path)


def analyze_event_distribution(df: pd.DataFrame) -> Dict:
    """Analyze distribution of events in the dataset."""
    # Count samples by event types
    event1_count = (df["event1"] == 1).sum()
    event2_count = (df["event2"] == 1).sum()
    censored_count = ((df["event1"] == 0) & (df["event2"] == 0)).sum()
    both_events_count = ((df["event1"] == 1) & (df["event2"] == 1)).sum()

    # Calculate proportions
    total = len(df)
    event1_prop = event1_count / total
    event2_prop = event2_count / total
    censored_prop = censored_count / total
    both_events_prop = both_events_count / total

    # Return summary
    return {
        "total_samples": total,
        "event1_count": event1_count,
        "event2_count": event2_count,
        "censored_count": censored_count,
        "both_events_count": both_events_count,
        "event1_proportion": event1_prop,
        "event2_proportion": event2_prop,
        "censored_proportion": censored_prop,
        "both_events_proportion": both_events_prop,
    }


def analyze_duration_distribution(df: pd.DataFrame) -> Dict:
    """Analyze distribution of durations in the dataset."""
    # Get statistical summaries
    duration1_stats = df["duration1"].describe()
    duration2_stats = df["duration2"].describe()

    # Plot histograms of durations
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(df["duration1"], bins=20, alpha=0.7)
    plt.title("Duration for Event 1")
    plt.xlabel("Duration")
    plt.ylabel("Count")

    plt.subplot(1, 2, 2)
    plt.hist(df["duration2"], bins=20, alpha=0.7)
    plt.title("Duration for Event 2")
    plt.xlabel("Duration")
    plt.ylabel("Count")

    plt.tight_layout()
    plt.savefig("hsa_duration_distribution.png")

    # Plot scatter of durations
    plt.figure(figsize=(8, 6))
    plt.scatter(df["duration1"], df["duration2"], alpha=0.5)
    plt.title("Duration1 vs Duration2")
    plt.xlabel("Duration1")
    plt.ylabel("Duration2")
    plt.savefig("hsa_duration_scatter.png")

    # Return summary
    return {"duration1_stats": duration1_stats, "duration2_stats": duration2_stats}


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


def prepare_sample_batch_from_df(
    df: pd.DataFrame, batch_size: int = 16
) -> Tuple[SAOutput, torch.Tensor]:
    """Prepare a batch of samples from the dataframe for testing with loss functions."""
    # Sample random rows from dataframe
    sample_df = df.sample(batch_size, random_state=42)

    # Convert to appropriate tensor format for loss functions
    # Each row is: [duration_percentile, event, fraction, duration] for each event
    # For num_events=2, shape will be [batch_size, 8]
    targets = torch.zeros(batch_size, 8)

    # Set event indicators and durations
    targets[:, 2] = torch.tensor(sample_df["event1"].values)
    targets[:, 3] = torch.tensor(sample_df["event2"].values)
    targets[:, 6] = torch.tensor(sample_df["duration1"].values)
    targets[:, 7] = torch.tensor(sample_df["duration2"].values)

    # Create simulated model predictions
    # We'll use random values but ensure they're consistent with the structure
    num_cuts = 10
    hazard = torch.rand(batch_size, 2, num_cuts)

    # Ensure survival is decreasing over time
    survival_base = torch.cumsum(
        torch.nn.functional.softplus(torch.randn(batch_size, 2, num_cuts)), dim=2
    )
    max_vals = survival_base.max(dim=2, keepdim=True)[0]
    survival_base = 1 - (survival_base / (max_vals + 1e-6))
    ones = torch.ones(batch_size, 2, 1)
    survival = torch.cat([ones, survival_base], dim=2)

    # Create fake logits
    logits = torch.zeros(batch_size, 2, num_cuts)

    # Create SAOutput
    predictions = SAOutput(logits=logits, hazard=hazard, survival=survival)

    return predictions, targets


def test_loss_functions_on_sample(
    df: pd.DataFrame, num_cuts: int = 10, batch_size: int = 16
) -> Dict:
    """Test loss functions on a sample batch from the HSA dataset."""
    # Create files needed for loss initialization
    duration_cuts_file = create_duration_cuts_file(num_cuts, 600)
    importance_weights_file = create_importance_weights_file(2)  # HSA has 2 events

    try:
        # Create loss instances
        multi_loss = MultiEventRankingLoss(
            duration_cuts=duration_cuts_file,
            importance_sample_weights=importance_weights_file,
            num_events=2,
            sigma=0.1,
            margin=0.05,
        )

        sample_loss = SampleRankingLoss(
            duration_cuts=duration_cuts_file,
            importance_sample_weights=importance_weights_file,
            num_events=2,
            sigma=0.1,
            margin=0.05,
        )

        # Prepare sample batch
        predictions, targets = prepare_sample_batch_from_df(df, batch_size)

        # Calculate losses
        multi_loss_val = multi_loss(predictions, targets)
        sample_loss_val = sample_loss(predictions, targets)

        # Calculate gradients
        hazard = predictions.hazard.clone().detach().requires_grad_(True)
        ones = torch.ones(batch_size, 2, 1)
        survival_base = (
            1 - torch.cumsum(torch.nn.functional.softplus(hazard), dim=2) / num_cuts
        )
        survival = torch.cat([ones, survival_base], dim=2)
        logits = torch.zeros_like(hazard)

        predictions_for_grad = SAOutput(logits=logits, hazard=hazard, survival=survival)

        # MultiEventRankingLoss gradient
        multi_loss_val = multi_loss(predictions_for_grad, targets)
        multi_loss_val.backward(retain_graph=True)
        multi_grad = hazard.grad.clone()
        hazard.grad.zero_()

        # SampleRankingLoss gradient
        sample_loss_val = sample_loss(predictions_for_grad, targets)
        sample_loss_val.backward()
        sample_grad = hazard.grad.clone()

        # Calculate gradient statistics
        multi_grad_norm = torch.norm(multi_grad).item()
        sample_grad_norm = torch.norm(sample_grad).item()

        # Calculate gradient sign match percentage
        sign_match = torch.sign(multi_grad) == torch.sign(sample_grad)
        sign_match_percentage = sign_match.float().mean().item() * 100

        # Calculate gradient correlation
        correlation = torch.corrcoef(
            torch.stack([multi_grad.flatten(), sample_grad.flatten()])
        )[0, 1].item()

        return {
            "multi_loss": multi_loss_val.item(),
            "sample_loss": sample_loss_val.item(),
            "loss_ratio": multi_loss_val.item() / sample_loss_val.item(),
            "multi_grad_norm": multi_grad_norm,
            "sample_grad_norm": sample_grad_norm,
            "grad_norm_ratio": multi_grad_norm / sample_grad_norm,
            "sign_match_percentage": sign_match_percentage,
            "gradient_correlation": correlation,
        }

    finally:
        # Clean up temporary files
        os.unlink(duration_cuts_file)
        os.unlink(importance_weights_file)


def analyze_competing_risks_structure(df: pd.DataFrame) -> None:
    """Analyze the competing risks structure of the HSA dataset."""
    # Check for samples with multiple events
    both_events = df[(df["event1"] == 1) & (df["event2"] == 1)]

    # Plot duration relationship for samples with both events
    if len(both_events) > 0:
        plt.figure(figsize=(8, 6))
        plt.scatter(both_events["duration1"], both_events["duration2"], alpha=0.7)
        plt.title("Duration Relationship for Samples with Both Events")
        plt.xlabel("Duration for Event 1")
        plt.ylabel("Duration for Event 2")
        plt.plot([0, 600], [0, 600], "r--")  # Diagonal line for reference
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.savefig("hsa_both_events_duration.png")

    # Analyze correlation between features and events
    feature_cols = [col for col in df.columns if col.startswith("x_")]

    # Calculate correlations
    corr_event1 = [df[col].corr(df["event1"]) for col in feature_cols]
    corr_event2 = [df[col].corr(df["event2"]) for col in feature_cols]

    # Plot correlations
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.bar(range(len(feature_cols)), corr_event1)
    plt.title("Correlation with Event 1")
    plt.xlabel("Feature Index")
    plt.ylabel("Correlation")
    plt.xticks(
        range(len(feature_cols)),
        [f"{i+1}" for i in range(len(feature_cols))],
        rotation=90,
    )

    plt.subplot(1, 2, 2)
    plt.bar(range(len(feature_cols)), corr_event2)
    plt.title("Correlation with Event 2")
    plt.xlabel("Feature Index")
    plt.ylabel("Correlation")
    plt.xticks(
        range(len(feature_cols)),
        [f"{i+1}" for i in range(len(feature_cols))],
        rotation=90,
    )

    plt.tight_layout()
    plt.savefig("hsa_feature_correlations.png")

    # Analyze if the events are correlated
    event_corr = df["event1"].corr(df["event2"])
    print(f"Correlation between events: {event_corr}")


def experiment_with_loss_parameters():
    """Run experiments with different loss parameters on HSA data."""
    # Load HSA dataset
    df = load_hsa_synthetic_data(
        "/Users/ddahlem/Documents/repos/open-disease-risk/sat/data/hsa-synthetic/simulated_data.csv"
    )

    # Create files needed for loss initialization
    duration_cuts_file = create_duration_cuts_file(10, 600)
    importance_weights_file = create_importance_weights_file(2)

    try:
        # Prepare sample batch
        predictions, targets = prepare_sample_batch_from_df(df, batch_size=32)

        # Test different sigma values
        sigma_values = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0]
        sigma_results = []

        print("\nTesting different sigma values:")
        for sigma in sigma_values:
            multi_loss = MultiEventRankingLoss(
                duration_cuts=duration_cuts_file,
                importance_sample_weights=importance_weights_file,
                num_events=2,
                sigma=sigma,
                margin=0.0,
            )

            sample_loss = SampleRankingLoss(
                duration_cuts=duration_cuts_file,
                importance_sample_weights=importance_weights_file,
                num_events=2,
                sigma=sigma,
                margin=0.0,
            )

            multi_loss_val = multi_loss(predictions, targets)
            sample_loss_val = sample_loss(predictions, targets)

            result = {
                "sigma": sigma,
                "multi_loss": multi_loss_val.item(),
                "sample_loss": sample_loss_val.item(),
                "ratio": multi_loss_val.item() / sample_loss_val.item(),
            }
            sigma_results.append(result)

            print(
                f"Sigma={sigma}: MultiEvent={multi_loss_val.item():.6f}, Sample={sample_loss_val.item():.6f}, Ratio={result['ratio']:.2f}"
            )

        # Test different margin values
        margin_values = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5]
        margin_results = []

        print("\nTesting different margin values:")
        for margin in margin_values:
            multi_loss = MultiEventRankingLoss(
                duration_cuts=duration_cuts_file,
                importance_sample_weights=importance_weights_file,
                num_events=2,
                sigma=0.1,
                margin=margin,
            )

            sample_loss = SampleRankingLoss(
                duration_cuts=duration_cuts_file,
                importance_sample_weights=importance_weights_file,
                num_events=2,
                sigma=0.1,
                margin=margin,
            )

            multi_loss_val = multi_loss(predictions, targets)
            sample_loss_val = sample_loss(predictions, targets)

            result = {
                "margin": margin,
                "multi_loss": multi_loss_val.item(),
                "sample_loss": sample_loss_val.item(),
                "ratio": multi_loss_val.item() / sample_loss_val.item(),
            }
            margin_results.append(result)

            print(
                f"Margin={margin}: MultiEvent={multi_loss_val.item():.6f}, Sample={sample_loss_val.item():.6f}, Ratio={result['ratio']:.2f}"
            )

        # Plot sigma results
        plt.figure(figsize=(10, 6))
        plt.plot(
            [r["sigma"] for r in sigma_results],
            [r["multi_loss"] for r in sigma_results],
            "b-o",
            label="MultiEventRankingLoss",
        )
        plt.plot(
            [r["sigma"] for r in sigma_results],
            [r["sample_loss"] for r in sigma_results],
            "r-o",
            label="SampleRankingLoss",
        )
        plt.xscale("log")
        plt.yscale("log")
        plt.title("Loss Value vs Sigma")
        plt.xlabel("Sigma")
        plt.ylabel("Loss Value")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend()
        plt.savefig("hsa_sigma_experiment.png")

        # Plot margin results
        plt.figure(figsize=(10, 6))
        plt.plot(
            [r["margin"] for r in margin_results],
            [r["multi_loss"] for r in margin_results],
            "b-o",
            label="MultiEventRankingLoss",
        )
        plt.plot(
            [r["margin"] for r in margin_results],
            [r["sample_loss"] for r in margin_results],
            "r-o",
            label="SampleRankingLoss",
        )
        plt.title("Loss Value vs Margin")
        plt.xlabel("Margin")
        plt.ylabel("Loss Value")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend()
        plt.savefig("hsa_margin_experiment.png")

    finally:
        # Clean up temporary files
        os.unlink(duration_cuts_file)
        os.unlink(importance_weights_file)


def main():
    """Run analysis of HSA synthetic dataset."""
    # Load HSA dataset
    print("Loading HSA synthetic dataset...")
    df = load_hsa_synthetic_data(
        "/Users/ddahlem/Documents/repos/open-disease-risk/sat/data/hsa-synthetic/simulated_data.csv"
    )

    # Analyze event distribution
    print("\nAnalyzing event distribution...")
    event_dist = analyze_event_distribution(df)
    for key, value in event_dist.items():
        print(f"{key}: {value}")

    # Analyze duration distribution
    print("\nAnalyzing duration distribution...")
    duration_dist = analyze_duration_distribution(df)
    print("Duration 1 statistics:")
    print(duration_dist["duration1_stats"])
    print("\nDuration 2 statistics:")
    print(duration_dist["duration2_stats"])

    # Analyze competing risks structure
    print("\nAnalyzing competing risks structure...")
    analyze_competing_risks_structure(df)

    # Test loss functions on sample
    print("\nTesting loss functions on sample batch...")
    loss_results = test_loss_functions_on_sample(df)
    for key, value in loss_results.items():
        print(f"{key}: {value}")

    # Run parameter experiments
    print("\nRunning parameter experiments...")
    experiment_with_loss_parameters()


if __name__ == "__main__":
    main()
