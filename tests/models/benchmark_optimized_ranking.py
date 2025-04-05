"""Benchmark tests for comparing original and vectorized ranking loss implementations"""

import argparse
import logging
import os
import tempfile
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# Configure minimal logging to avoid debug messages during benchmarks
logging.basicConfig(level=logging.WARNING)


def get_best_device():
    """Get the best available device for PyTorch computations."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif (
        hasattr(torch, "mps")
        and torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
    ):
        return torch.device("mps")
    else:
        return torch.device("cpu")


from sat.loss.ranking.multievent import MultiEventRankingLoss
from sat.loss.ranking.sample import SampleRankingLoss
from sat.models.heads.output import SAOutput


def create_synthetic_survival_data(
    batch_size=128, num_events=2, num_labels=30, device="cpu"
):
    """Create synthetic data for survival benchmarks with specific dimensions."""
    # Create input features
    sequence_output = torch.randn(batch_size, 64, device=device)

    # Create reference labels
    references = torch.zeros(batch_size, 4 * num_events, device=device)

    # Set duration percentiles
    references[:, 0:num_events] = torch.randint(
        0, num_labels - 1, (batch_size, num_events), device=device
    )

    # Set events (make ~20% have events)
    events = torch.zeros(batch_size, num_events, device=device)
    for event_idx in range(num_events):
        mask = torch.rand(batch_size, device=device) < 0.2
        events[mask, event_idx] = 1

    # Ensure no sample has multiple event types if num_events > 1
    if num_events > 1:
        multiple_events = torch.sum(events, dim=1) > 1
        if multiple_events.any():
            for i in range(batch_size):
                if multiple_events[i]:
                    # Keep only the first event type
                    first_event = torch.argmax(events[i])
                    events[i] = torch.zeros(num_events, device=device)
                    events[i, first_event] = 1

    references[:, num_events : 2 * num_events] = events

    # Set fraction with quantile
    references[:, 2 * num_events : 3 * num_events] = torch.rand(
        batch_size, num_events, device=device
    )

    # Set durations
    references[:, 3 * num_events : 4 * num_events] = (
        torch.rand(batch_size, num_events, device=device) * 100
    )

    # Create hazard, risk, and survival
    hazard = torch.rand(batch_size, num_events, num_labels + 1, device=device) * 0.1
    risk = torch.cumsum(hazard, dim=2)
    risk = torch.clamp(risk, max=1.0)
    survival = 1.0 - risk

    # Create logits
    logits = torch.log(hazard + 1e-8)

    # Create SAOutput
    output = SAOutput(
        loss=None,
        logits=logits,
        hazard=hazard,
        risk=risk,
        survival=survival,
        hidden_states=sequence_output,
    )

    # Create duration cuts
    duration_cuts = torch.linspace(0, 100, steps=num_labels, device=device).tolist()

    return {
        "sequence_output": sequence_output,
        "references": references,
        "output": output,
        "duration_cuts": duration_cuts,
        "num_features": 64,
        "num_events": num_events,
        "num_labels": num_labels,
        "batch_size": batch_size,
    }


def run_ranking_loss_benchmark(data, loss_fn, num_trials=5, tag="", verbose=True):
    """Run benchmark for a specific loss function and return results."""
    output = data["output"]
    references = data["references"]
    device = references.device

    # Warm-up run
    with torch.no_grad():
        _ = loss_fn(output, references)

    # Device synchronization function
    def sync_device():
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()

    # Benchmark forward pass
    sync_device()
    forward_times = []
    start_time = time.time()
    for _ in range(num_trials):
        with torch.no_grad():
            loss = loss_fn(output, references)
            loss_value = loss.item()
        sync_device()
        forward_times.append(time.time() - start_time)
        start_time = time.time()
    forward_time = sum(forward_times) / len(forward_times)

    # Benchmark backward pass
    backward_times = []
    for _ in range(num_trials):
        # Create a copy of output with requires_grad=True for tensors
        output_copy = SAOutput(
            loss=output.loss,
            logits=output.logits.detach().clone().requires_grad_(True),
            hazard=output.hazard.detach().clone().requires_grad_(True),
            risk=output.risk.detach().clone().requires_grad_(True),
            survival=output.survival.detach().clone().requires_grad_(True),
            hidden_states=output.hidden_states.detach().clone().requires_grad_(True),
        )

        sync_device()
        start_time = time.time()
        try:
            loss = loss_fn(output_copy, references)
            if loss.requires_grad:
                loss.backward()
            sync_device()
            backward_times.append(time.time() - start_time)
        except Exception as e:
            print(f"Backward pass error with {tag}: {e}")
            backward_times.append(float("nan"))

    backward_time = np.nanmean(backward_times)

    # Calculate total time (forward + backward)
    total_time = forward_time + backward_time

    if verbose:
        print(
            f"{tag} - Forward: {forward_time:.6f}s, "
            f"Backward: {backward_time:.6f}s, "
            f"Total: {total_time:.6f}s, "
            f"Value: {loss_value:.6f}"
        )

    return {
        "forward_time": forward_time,
        "backward_time": backward_time,
        "total_time": total_time,
        "output_value": loss_value,
        "tag": tag,
    }


def compare_loss_implementations(
    batch_size=128, num_events=2, num_trials=5, device="cpu", verbose=True
):
    """Compare the performance of original and vectorized ranking loss implementations."""
    print(
        f"\nComparing implementations with batch_size={batch_size}, num_events={num_events} on {device}"
    )

    # Create synthetic data
    data = create_synthetic_survival_data(
        batch_size=batch_size, num_events=num_events, num_labels=30, device=device
    )

    # Create temp file for duration cuts
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
        cuts_file = f.name
        pd.DataFrame(data["duration_cuts"]).to_csv(cuts_file, index=False, header=False)

    try:
        # Subclass with original implementation for testing
        # Add the original ranking_loss implementation for testing
        def original_ranking_loss(self, events, durations, survival, hazard, weights):
            """
            Original implementation of ranking loss for comparison purposes.
            This is the unoptimized version that was used before optimizations.
            """
            n = events.shape[0]  # batch size
            e = events.shape[1]  # number of events
            device = events.device

            # Create tensor for censored indicator (n x e)
            I = events.to(bool)
            I_censored = ~I

            # Initialize tensor for durations cut points (n x e x tn)
            T = self.duration_cuts.to(device)
            T = T.expand(n, e, -1)

            # Calculate the index for time bins
            durations_expanded = durations.unsqueeze(2)  # (n x e x 1)
            cuts_expanded = self.duration_cuts.to(device).view(1, 1, -1)  # (1 x 1 x tn)
            indexSmaller = cuts_expanded <= durations_expanded  # (n x e x tn)

            t0Index = torch.sum(indexSmaller, dim=2) - 1  # (n x e)

            # Create survival at time t for each event
            SatT = torch.zeros(n, e, e, device=device)
            t_epsilon = (
                self.duration_cuts[-1] - self.duration_cuts[0]
            ) / self.duration_cuts[-1]
            TMinus = torch.zeros(n, e, e, device=device)
            SatTMinus = torch.zeros(n, e, e, device=device)

            # Loop through events (i) and observations (j)
            for i in range(n):
                for j in range(e):
                    # Find the interval for this observation and event
                    # e.g., for t = 5.2, if we have cuts at [1, 5, 10],
                    # t0Index[i, j] = 1 (position of 5 in the array)
                    t0_idx = t0Index[i, j]
                    t1_idx = t0_idx + 1

                    # Handle boundary cases
                    if t0_idx < 0:
                        t0_idx = 0
                    if t1_idx >= len(self.duration_cuts):
                        t1_idx = len(self.duration_cuts) - 1

                    # Get the time points
                    t0 = T[i, j, t0_idx]
                    t1 = T[i, j, t1_idx]

                    # Get survival values at these time points
                    S_t0 = survival[i, j, t0_idx]
                    S_t1 = survival[i, j, t1_idx]

                    # Interpolate hazard rate
                    dT = t1 - t0
                    h_star = hazard[i, j, t0_idx]

                    if dT > 0:
                        # Add small epsilon for numerical stability
                        h_star = (torch.log(S_t0 + 1e-6) - torch.log(S_t1 + 1e-6)) / dT

                    # Calculate survival at this specific time point
                    for k in range(e):
                        # Survival at time t
                        SatT[i, j, k] = S_t0 * torch.exp(
                            -(durations[i, j] - t0) * h_star
                        )

                        # For each event, calculate a time t - epsilon
                        TMinus[i, j, k] = durations[i, j] - t_epsilon
                        TMinus[i, j, k] = torch.max(
                            TMinus[i, j, k], torch.tensor(0.0, device=device)
                        )

                        # Calculate survival at t - epsilon
                        SatTMinus[i, j, k] = S_t0 * torch.exp(
                            -(TMinus[i, j, k] - t0) * h_star
                        )

            # Calculate ranking pairs for each observation
            dS1 = torch.zeros(n, e, e, device=device)
            dS2 = torch.zeros(n, e, e, device=device)
            dS3 = torch.zeros(n, e, e, device=device)

            for i in range(n):
                for j in range(e):  # First event
                    for k in range(e):  # Second event
                        # Calculate the difference in durations
                        if durations[i, j] > durations[i, k]:
                            # For each comparison case
                            dS1[i, j, k] = SatT[i, j, j] - SatT[i, k, j]  # Case 1

                            if events[i, j] == 1 and events[i, k] == 1:
                                dS2[i, j, k] = (
                                    SatTMinus[i, j, k] - SatTMinus[i, k, k]
                                )  # Case 2

                            if events[i, j] == 1 and events[i, k] == 0:
                                dS3[i, j, k] = SatT[i, j, j] - SatT[i, k, j]  # Case 3

            # Calculate mask tensors
            A1 = torch.zeros(n, e, e, device=device)
            A2 = torch.zeros(n, e, e, device=device)
            A3 = torch.zeros(n, e, e, device=device)

            for i in range(n):
                for j in range(e):
                    for k in range(e):
                        if durations[i, j] > durations[i, k]:
                            # Case 1: Any event (censored or not)
                            if events[i, j] == 1:
                                A1[i, j, k] = 1

                                # Case 2: Both had events
                                if events[i, k] == 1:
                                    A2[i, j, k] = 1

                                # Case 3: First had event, second was censored
                                if events[i, k] == 0:
                                    A3[i, j, k] = 1

            # Calcuate margin penalties if applicable
            if hasattr(self, "margin") and self.margin > 0:
                margin_dS1 = torch.clamp(self.margin - dS1, min=0.0) * A1
                margin_dS2 = torch.clamp(self.margin - dS2, min=0.0) * A2
                margin_dS3 = torch.clamp(self.margin - dS3, min=0.0) * A3

                # Loss with margin
                loss_dS1 = torch.exp(dS1 / self.sigma) + margin_dS1
                loss_dS2 = torch.exp(dS2 / self.sigma) + margin_dS2
                loss_dS3 = torch.exp(dS3 / self.sigma) + margin_dS3
            else:
                # Traditional loss without margin
                loss_dS1 = torch.exp(dS1 / self.sigma)
                loss_dS2 = torch.exp(dS2 / self.sigma)
                loss_dS3 = torch.exp(dS3 / self.sigma)

            # Apply weights if provided
            if weights is not None:
                weights = weights.to(device)
                loss_term = weights * (A1 * loss_dS1 + A2 * loss_dS2 + A3 * loss_dS3)
            else:
                loss_term = A1 * loss_dS1 + A2 * loss_dS2 + A3 * loss_dS3

            # Calculate number of non-zero elements for proper normalization
            num_valid = torch.sum((A1 + A2 + A3) > 0).item()

            if num_valid > 0:
                eta = torch.sum(loss_term) / num_valid
            else:
                # Return zero tensor with gradient if no valid comparisons
                eta = torch.tensor(0.0, device=device, requires_grad=True)

            return eta

        class OriginalSampleRankingLoss(SampleRankingLoss):
            def forward(self, predictions, references):
                # Permute the dimensions to change from [batch, events] to [events, batch]
                events = self.events(references).permute(1, 0)
                e = events.shape[1]  # Batch size after permutation

                # Create weight tensor if needed - permuted to match the new tensor orientation
                weights_expanded = None
                if self.weights is not None:
                    # Skip the first weight (censoring) and use only event weights
                    weights_expanded = self.weights[1:].to(references.device)
                    # Expand to match the expected dimensions with the permuted orientation
                    weights_expanded = (
                        weights_expanded.unsqueeze(1).unsqueeze(2).repeat(1, e, e)
                    )

                # Use the original implementation
                eta = original_ranking_loss(
                    self,
                    events,
                    self.durations(references).permute(1, 0),
                    predictions.survival.permute(1, 0, 2),
                    predictions.hazard.permute(1, 0, 2),
                    weights_expanded,
                )
                return eta

        class OriginalMultiEventRankingLoss(MultiEventRankingLoss):
            def forward(self, predictions, references):
                # Extract events and set up dimensions - no permutation needed
                events = self.events(references)
                n = events.shape[0]  # Batch size
                e = events.shape[1]  # Number of events
                device = references.device

                # For efficiency, only calculate weights expansion once and store
                weights_expanded = None
                if self.weights is not None:
                    # Skip the first weight (censoring) and use only event weights
                    weights_expanded = self.weights[1:].to(device)
                    # Expand to match the expected dimensions
                    weights_expanded = (
                        weights_expanded.unsqueeze(0)
                        .unsqueeze(2)
                        .repeat(1, 1, e)
                        .expand(n, -1, -1)
                    )

                # Use the original implementation
                eta = original_ranking_loss(
                    self,
                    events,
                    self.durations(references),
                    predictions.survival,
                    predictions.hazard,
                    weights_expanded,
                )
                return eta

        # Create loss functions
        loss_functions = [
            # Sample Ranking Loss implementations - original implementation
            (
                OriginalSampleRankingLoss(
                    num_events=num_events,
                    duration_cuts=cuts_file,
                    sigma=0.1,
                    margin=0.0,
                ),
                "SampleRankingLoss (Original)",
            ),
            # Sample Ranking Loss - vectorized implementation
            (
                SampleRankingLoss(
                    num_events=num_events,
                    duration_cuts=cuts_file,
                    sigma=0.1,
                    margin=0.0,
                ),
                "SampleRankingLoss (Vectorized)",
            ),
            # Multi-Event Ranking Loss - original implementation
            (
                OriginalMultiEventRankingLoss(
                    num_events=num_events,
                    duration_cuts=cuts_file,
                    sigma=0.1,
                    margin=0.0,
                ),
                "MultiEventRankingLoss (Original)",
            ),
            # Multi-Event Ranking Loss - vectorized implementation
            (
                MultiEventRankingLoss(
                    num_events=num_events,
                    duration_cuts=cuts_file,
                    sigma=0.1,
                    margin=0.0,
                ),
                "MultiEventRankingLoss (Vectorized)",
            ),
        ]

        results = []

        # Run benchmarks for each loss function
        for loss_fn, tag in loss_functions:
            result = run_ranking_loss_benchmark(
                data, loss_fn, num_trials=num_trials, tag=tag, verbose=verbose
            )
            results.append(result)

        return results

    finally:
        # Clean up temp file
        if os.path.exists(cuts_file):
            os.remove(cuts_file)


def run_comprehensive_benchmark(
    batch_sizes=[32, 128, 512], event_counts=[1, 2, 5], num_trials=5, device="cpu"
):
    """Run a comprehensive benchmark comparing original vs vectorized implementations across multiple configurations."""
    print(
        f"Running comprehensive benchmark on {device} with {num_trials} trials per configuration"
    )

    all_results = []

    # Run benchmarks for each configuration
    for batch_size in batch_sizes:
        for num_events in event_counts:
            results = compare_loss_implementations(
                batch_size=batch_size,
                num_events=num_events,
                num_trials=num_trials,
                device=device,
                verbose=True,
            )

            for result in results:
                result["batch_size"] = batch_size
                result["num_events"] = num_events
                all_results.append(result)

    return all_results


def create_summary_tables(results):
    """Create summary tables from benchmark results."""
    # Convert results to DataFrame
    df = pd.DataFrame(results)

    # Create a table for forward time
    forward_table = df.pivot_table(
        index=["tag", "num_events"],
        columns=["batch_size"],
        values=["forward_time"],
        aggfunc="mean",
    )

    # Create a table for backward time
    backward_table = df.pivot_table(
        index=["tag", "num_events"],
        columns=["batch_size"],
        values=["backward_time"],
        aggfunc="mean",
    )

    # Create a table for total time
    total_table = df.pivot_table(
        index=["tag", "num_events"],
        columns=["batch_size"],
        values=["total_time"],
        aggfunc="mean",
    )

    return {
        "forward_table": forward_table,
        "backward_table": backward_table,
        "total_table": total_table,
        "raw_df": df,
    }


def calculate_speedup(results):
    """Calculate speedup factors between original and vectorized implementations."""
    df = pd.DataFrame(results)

    # Separate original and vectorized results
    original_sample = df[df["tag"] == "SampleRankingLoss (Original)"]
    vectorized_sample = df[df["tag"] == "SampleRankingLoss (Vectorized)"]
    original_multievent = df[df["tag"] == "MultiEventRankingLoss (Original)"]
    vectorized_multievent = df[df["tag"] == "MultiEventRankingLoss (Vectorized)"]

    # Create DataFrames for speedup calculations
    sample_speedup = []
    multievent_speedup = []

    # Group by batch_size and num_events
    groups = df.groupby(["batch_size", "num_events"])

    for (batch_size, num_events), group in groups:
        # Calculate Sample ranking speedup
        orig_sample = group[group["tag"] == "SampleRankingLoss (Original)"]
        opt_sample = group[group["tag"] == "SampleRankingLoss (Vectorized)"]

        if not orig_sample.empty and not opt_sample.empty:
            sample_speedup.append(
                {
                    "batch_size": batch_size,
                    "num_events": num_events,
                    "forward_speedup": orig_sample["forward_time"].values[0]
                    / opt_sample["forward_time"].values[0],
                    "backward_speedup": orig_sample["backward_time"].values[0]
                    / opt_sample["backward_time"].values[0],
                    "total_speedup": orig_sample["total_time"].values[0]
                    / opt_sample["total_time"].values[0],
                }
            )

        # Calculate MultiEvent ranking speedup
        orig_multi = group[group["tag"] == "MultiEventRankingLoss (Original)"]
        opt_multi = group[group["tag"] == "MultiEventRankingLoss (Vectorized)"]

        if not orig_multi.empty and not opt_multi.empty:
            multievent_speedup.append(
                {
                    "batch_size": batch_size,
                    "num_events": num_events,
                    "forward_speedup": orig_multi["forward_time"].values[0]
                    / opt_multi["forward_time"].values[0],
                    "backward_speedup": orig_multi["backward_time"].values[0]
                    / opt_multi["backward_time"].values[0],
                    "total_speedup": orig_multi["total_time"].values[0]
                    / opt_multi["total_time"].values[0],
                }
            )

    return {
        "sample_speedup": (
            pd.DataFrame(sample_speedup) if sample_speedup else pd.DataFrame()
        ),
        "multievent_speedup": (
            pd.DataFrame(multievent_speedup) if multievent_speedup else pd.DataFrame()
        ),
    }


def visualize_results(results, output_dir="logs"):
    """Create visualizations of benchmark results."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df = pd.DataFrame(results)
    batch_sizes = sorted(df["batch_size"].unique())
    event_counts = sorted(df["num_events"].unique())
    implementation_tags = df["tag"].unique()

    # Plotting time comparison by implementation
    if len(event_counts) == 1:
        # Special case for single event count
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        num_events = event_counts[0]
        event_df = df[df["num_events"] == num_events]

        # Forward time plot
        ax = axes[0]
        for tag in implementation_tags:
            tag_df = event_df[event_df["tag"] == tag]
            if not tag_df.empty:
                ax.plot(tag_df["batch_size"], tag_df["forward_time"], "o-", label=tag)

        ax.set_title(f"Forward Time (Events={num_events})")
        ax.set_xlabel("Batch Size")
        ax.set_ylabel("Time (seconds)")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.7)

        # Backward time plot
        ax = axes[1]
        for tag in implementation_tags:
            tag_df = event_df[event_df["tag"] == tag]
            if not tag_df.empty:
                ax.plot(tag_df["batch_size"], tag_df["backward_time"], "o-", label=tag)

        ax.set_title(f"Backward Time (Events={num_events})")
        ax.set_xlabel("Batch Size")
        ax.set_ylabel("Time (seconds)")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.7)

        # Total time plot
        ax = axes[2]
        for tag in implementation_tags:
            tag_df = event_df[event_df["tag"] == tag]
            if not tag_df.empty:
                ax.plot(tag_df["batch_size"], tag_df["total_time"], "o-", label=tag)

        ax.set_title(f"Total Time (Events={num_events})")
        ax.set_xlabel("Batch Size")
        ax.set_ylabel("Time (seconds)")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.7)
    else:
        # Multiple event counts
        fig, axes = plt.subplots(
            len(event_counts), 3, figsize=(18, 5 * len(event_counts))
        )

        for i, num_events in enumerate(event_counts):
            event_df = df[df["num_events"] == num_events]

            # Forward time plot
            ax = axes[i, 0]
            for tag in implementation_tags:
                tag_df = event_df[event_df["tag"] == tag]
                if not tag_df.empty:
                    ax.plot(
                        tag_df["batch_size"], tag_df["forward_time"], "o-", label=tag
                    )

            ax.set_title(f"Forward Time (Events={num_events})")
            ax.set_xlabel("Batch Size")
            ax.set_ylabel("Time (seconds)")
            ax.legend()
            ax.grid(True, linestyle="--", alpha=0.7)

            # Backward time plot
            ax = axes[i, 1]
            for tag in implementation_tags:
                tag_df = event_df[event_df["tag"] == tag]
                if not tag_df.empty:
                    ax.plot(
                        tag_df["batch_size"], tag_df["backward_time"], "o-", label=tag
                    )

            ax.set_title(f"Backward Time (Events={num_events})")
            ax.set_xlabel("Batch Size")
            ax.set_ylabel("Time (seconds)")
            ax.legend()
            ax.grid(True, linestyle="--", alpha=0.7)

            # Total time plot
            ax = axes[i, 2]
            for tag in implementation_tags:
                tag_df = event_df[event_df["tag"] == tag]
                if not tag_df.empty:
                    ax.plot(tag_df["batch_size"], tag_df["total_time"], "o-", label=tag)

            ax.set_title(f"Total Time (Events={num_events})")
            ax.set_xlabel("Batch Size")
            ax.set_ylabel("Time (seconds)")
            ax.legend()
            ax.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ranking_time_comparison.png"))
    plt.close()

    # Plot speedup factors if we have the data
    speedup = calculate_speedup(results)
    event_markers = ["o", "s", "^", "D", "v"]

    # Visualize Sample ranking speedup
    if "sample_speedup" in speedup and not speedup["sample_speedup"].empty:
        sample_df = speedup["sample_speedup"]

        # Skip if we don't have the necessary columns
        if not (
            "batch_size" in sample_df.columns
            and "num_events" in sample_df.columns
            and any(
                x in sample_df.columns
                for x in ["forward_speedup", "backward_speedup", "total_speedup"]
            )
        ):
            return True

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Forward speedup
        ax = axes[0]
        if "forward_speedup" in sample_df.columns:
            for i, num_events in enumerate(event_counts):
                event_df = (
                    sample_df[sample_df["num_events"] == num_events]
                    if "num_events" in sample_df.columns
                    else sample_df
                )
                if not event_df.empty:
                    marker = event_markers[i % len(event_markers)]
                    ax.plot(
                        event_df["batch_size"],
                        event_df["forward_speedup"],
                        marker + "-",
                        label=f"Events={num_events}",
                    )

            ax.set_title("SampleRankingLoss Forward Speedup")
            ax.set_xlabel("Batch Size")
            ax.set_ylabel("Speedup Factor (Original/Vectorized)")
            ax.axhline(y=1.0, color="r", linestyle="--", alpha=0.5)
            ax.legend()
            ax.grid(True, linestyle="--", alpha=0.7)

        # Backward speedup
        ax = axes[1]
        if "backward_speedup" in sample_df.columns:
            for i, num_events in enumerate(event_counts):
                event_df = (
                    sample_df[sample_df["num_events"] == num_events]
                    if "num_events" in sample_df.columns
                    else sample_df
                )
                if not event_df.empty:
                    marker = event_markers[i % len(event_markers)]
                    ax.plot(
                        event_df["batch_size"],
                        event_df["backward_speedup"],
                        marker + "-",
                        label=f"Events={num_events}",
                    )

            ax.set_title("SampleRankingLoss Backward Speedup")
            ax.set_xlabel("Batch Size")
            ax.set_ylabel("Speedup Factor (Original/Vectorized)")
            ax.axhline(y=1.0, color="r", linestyle="--", alpha=0.5)
            ax.legend()
            ax.grid(True, linestyle="--", alpha=0.7)

        # Total speedup
        ax = axes[2]
        if "total_speedup" in sample_df.columns:
            for i, num_events in enumerate(event_counts):
                event_df = (
                    sample_df[sample_df["num_events"] == num_events]
                    if "num_events" in sample_df.columns
                    else sample_df
                )
                if not event_df.empty:
                    marker = event_markers[i % len(event_markers)]
                    ax.plot(
                        event_df["batch_size"],
                        event_df["total_speedup"],
                        marker + "-",
                        label=f"Events={num_events}",
                    )

            ax.set_title("SampleRankingLoss Total Speedup")
            ax.set_xlabel("Batch Size")
            ax.set_ylabel("Speedup Factor (Original/Vectorized)")
            ax.axhline(y=1.0, color="r", linestyle="--", alpha=0.5)
            ax.legend()
            ax.grid(True, linestyle="--", alpha=0.7)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "sample_ranking_speedup.png"))
        plt.close()

    # Visualize MultiEvent ranking speedup
    if "multievent_speedup" in speedup and not speedup["multievent_speedup"].empty:
        multi_df = speedup["multievent_speedup"]

        # Skip if we don't have the necessary columns
        if not (
            "batch_size" in multi_df.columns
            and "num_events" in multi_df.columns
            and any(
                x in multi_df.columns
                for x in ["forward_speedup", "backward_speedup", "total_speedup"]
            )
        ):
            return True

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Forward speedup
        ax = axes[0]
        if "forward_speedup" in multi_df.columns:
            for i, num_events in enumerate(event_counts):
                event_df = (
                    multi_df[multi_df["num_events"] == num_events]
                    if "num_events" in multi_df.columns
                    else multi_df
                )
                if not event_df.empty:
                    marker = event_markers[i % len(event_markers)]
                    ax.plot(
                        event_df["batch_size"],
                        event_df["forward_speedup"],
                        marker + "-",
                        label=f"Events={num_events}",
                    )

            ax.set_title("MultiEventRankingLoss Forward Speedup")
            ax.set_xlabel("Batch Size")
            ax.set_ylabel("Speedup Factor (Original/Vectorized)")
            ax.axhline(y=1.0, color="r", linestyle="--", alpha=0.5)
            ax.legend()
            ax.grid(True, linestyle="--", alpha=0.7)

        # Backward speedup
        ax = axes[1]
        if "backward_speedup" in multi_df.columns:
            for i, num_events in enumerate(event_counts):
                event_df = (
                    multi_df[multi_df["num_events"] == num_events]
                    if "num_events" in multi_df.columns
                    else multi_df
                )
                if not event_df.empty:
                    marker = event_markers[i % len(event_markers)]
                    ax.plot(
                        event_df["batch_size"],
                        event_df["backward_speedup"],
                        marker + "-",
                        label=f"Events={num_events}",
                    )

            ax.set_title("MultiEventRankingLoss Backward Speedup")
            ax.set_xlabel("Batch Size")
            ax.set_ylabel("Speedup Factor (Original/Vectorized)")
            ax.axhline(y=1.0, color="r", linestyle="--", alpha=0.5)
            ax.legend()
            ax.grid(True, linestyle="--", alpha=0.7)

        # Total speedup
        ax = axes[2]
        if "total_speedup" in multi_df.columns:
            for i, num_events in enumerate(event_counts):
                event_df = (
                    multi_df[multi_df["num_events"] == num_events]
                    if "num_events" in multi_df.columns
                    else multi_df
                )
                if not event_df.empty:
                    marker = event_markers[i % len(event_markers)]
                    ax.plot(
                        event_df["batch_size"],
                        event_df["total_speedup"],
                        marker + "-",
                        label=f"Events={num_events}",
                    )

            ax.set_title("MultiEventRankingLoss Total Speedup")
            ax.set_xlabel("Batch Size")
            ax.set_ylabel("Speedup Factor (Original/Vectorized)")
            ax.axhline(y=1.0, color="r", linestyle="--", alpha=0.5)
            ax.legend()
            ax.grid(True, linestyle="--", alpha=0.7)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "multievent_ranking_speedup.png"))
        plt.close()

    # Only create heatmaps if there are multiple batch sizes and event counts
    if len(batch_sizes) > 1 and len(event_counts) > 1:
        # Check that we have the required data frames
        if (
            "sample_speedup" in speedup
            and not speedup["sample_speedup"].empty
            and "multievent_speedup" in speedup
            and not speedup["multievent_speedup"].empty
        ):
            # Check that both dataframes have the necessary columns for creating the heatmap
            if set(["batch_size", "num_events", "total_speedup"]).issubset(
                speedup["sample_speedup"].columns
            ) and set(["batch_size", "num_events", "total_speedup"]).issubset(
                speedup["multievent_speedup"].columns
            ):

                fig, axes = plt.subplots(2, 3, figsize=(18, 12))

                # Process data for heatmaps
                for df_name, df, row_idx in [
                    ("SampleRankingLoss", speedup["sample_speedup"], 0),
                    ("MultiEventRankingLoss", speedup["multievent_speedup"], 1),
                ]:
                    if df.empty:
                        continue

                    # Create pivot tables
                    forward_pivot = df.pivot_table(
                        index="num_events",
                        columns="batch_size",
                        values="forward_speedup",
                        aggfunc="mean",
                    )
                    backward_pivot = df.pivot_table(
                        index="num_events",
                        columns="batch_size",
                        values="backward_speedup",
                        aggfunc="mean",
                    )
                    total_pivot = df.pivot_table(
                        index="num_events",
                        columns="batch_size",
                        values="total_speedup",
                        aggfunc="mean",
                    )

                    # Plot heatmaps
                    for pivot, title, col_idx in [
                        (forward_pivot, f"{df_name} Forward Speedup", 0),
                        (backward_pivot, f"{df_name} Backward Speedup", 1),
                        (total_pivot, f"{df_name} Total Speedup", 2),
                    ]:
                        ax = axes[row_idx, col_idx]
                        im = ax.imshow(pivot, cmap="viridis")

                        # Add colorbar
                        fig.colorbar(im, ax=ax)

                        # Add labels
                        ax.set_title(title)
                        ax.set_yticks(np.arange(len(pivot.index)))
                        ax.set_yticklabels(pivot.index)
                        ax.set_xticks(np.arange(len(pivot.columns)))
                        ax.set_xticklabels(pivot.columns)
                        ax.set_xlabel("Batch Size")
                        ax.set_ylabel("Number of Events")

                        # Add text annotations
                        for i in range(len(pivot.index)):
                            for j in range(len(pivot.columns)):
                                if not np.isnan(pivot.iloc[i, j]):
                                    ax.text(
                                        j,
                                        i,
                                        f"{pivot.iloc[i, j]:.2f}",
                                        ha="center",
                                        va="center",
                                        color="w",
                                    )

                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, "speedup_heatmap.png"))
                plt.close()

    return True


def save_results(results, output_dir="logs"):
    """Save benchmark results to CSV and generate a summary markdown file."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save raw results
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, "ranking_benchmark_results.csv"))

    # Create and save summary tables
    tables = create_summary_tables(results)
    speedup = calculate_speedup(results)

    # Save individual tables
    tables["forward_table"].to_csv(os.path.join(output_dir, "forward_time_table.csv"))
    tables["backward_table"].to_csv(os.path.join(output_dir, "backward_time_table.csv"))
    tables["total_table"].to_csv(os.path.join(output_dir, "total_time_table.csv"))

    # Only save speedup data if available
    if "sample_speedup" in speedup and not speedup["sample_speedup"].empty:
        speedup["sample_speedup"].to_csv(
            os.path.join(output_dir, "sample_ranking_speedup.csv")
        )

    if "multievent_speedup" in speedup and not speedup["multievent_speedup"].empty:
        speedup["multievent_speedup"].to_csv(
            os.path.join(output_dir, "multievent_ranking_speedup.csv")
        )

    # Create summary markdown file
    with open(os.path.join(output_dir, "ranking_benchmark_summary.md"), "w") as f:
        f.write("# Ranking Loss Benchmark Results\n\n")

        f.write("## Forward Pass Time (seconds)\n\n")
        f.write(tables["forward_table"].to_markdown())
        f.write("\n\n")

        f.write("## Backward Pass Time (seconds)\n\n")
        f.write(tables["backward_table"].to_markdown())
        f.write("\n\n")

        f.write("## Total Time (seconds)\n\n")
        f.write(tables["total_table"].to_markdown())
        f.write("\n\n")

        # Only include speedup sections if data is available
        if "sample_speedup" in speedup and not speedup["sample_speedup"].empty:
            f.write("## SampleRankingLoss Speedup Factors\n\n")
            f.write(speedup["sample_speedup"].to_markdown())
            f.write("\n\n")

        if "multievent_speedup" in speedup and not speedup["multievent_speedup"].empty:
            f.write("## MultiEventRankingLoss Speedup Factors\n\n")
            f.write(speedup["multievent_speedup"].to_markdown())
            f.write("\n\n")

        # Calculate average speedup (safely)
        f.write("### Average Speedup\n\n")

        sample_avg = None
        if "sample_speedup" in speedup and not speedup["sample_speedup"].empty:
            if "total_speedup" in speedup["sample_speedup"].columns:
                sample_avg = speedup["sample_speedup"]["total_speedup"].mean()
                f.write(f"- SampleRankingLoss: {sample_avg:.2f}x\n")
            else:
                f.write("- SampleRankingLoss: N/A\n")
        else:
            f.write("- SampleRankingLoss: N/A\n")

        multi_avg = None
        if "multievent_speedup" in speedup and not speedup["multievent_speedup"].empty:
            if "total_speedup" in speedup["multievent_speedup"].columns:
                multi_avg = speedup["multievent_speedup"]["total_speedup"].mean()
                f.write(f"- MultiEventRankingLoss: {multi_avg:.2f}x\n")
            else:
                f.write("- MultiEventRankingLoss: N/A\n")
        else:
            f.write("- MultiEventRankingLoss: N/A\n")

        f.write("\n")

        f.write("## Visualization\n\n")
        f.write(
            "See the generated PNG files in the same directory for visualizations of these results.\n"
        )

    return True


def main():
    """Main function to run the benchmark."""
    parser = argparse.ArgumentParser(
        description="Benchmark optimized ranking loss functions"
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[32, 128, 512],
        help="Batch sizes to test",
    )
    parser.add_argument(
        "--event-counts",
        type=int,
        nargs="+",
        default=[1, 2, 5],
        help="Number of events to test",
    )
    parser.add_argument(
        "--num-trials", type=int, default=5, help="Number of trials per configuration"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to run on (auto, cpu, cuda, or mps)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="logs", help="Directory to save results"
    )

    args = parser.parse_args()

    # Handle device selection
    if args.device == "auto":
        device = get_best_device()
        args.device = device.type
        print(f"Auto-selected device: {args.device}")
    else:
        # Check device availability
        if args.device == "cuda" and not torch.cuda.is_available():
            print("CUDA requested but not available. Falling back to CPU.")
            args.device = "cpu"
        elif args.device == "mps" and not (
            torch.backends.mps.is_available() and torch.backends.mps.is_built()
        ):
            print("MPS requested but not available. Falling back to CPU.")
            args.device = "cpu"

    print(f"Starting benchmark with the following configuration:")
    print(f"  - Batch sizes: {args.batch_sizes}")
    print(f"  - Event counts: {args.event_counts}")
    print(f"  - Trials per config: {args.num_trials}")
    print(f"  - Device: {args.device}")
    print(f"  - Output directory: {args.output_dir}")

    # Run the benchmark
    results = run_comprehensive_benchmark(
        batch_sizes=args.batch_sizes,
        event_counts=args.event_counts,
        num_trials=args.num_trials,
        device=args.device,
    )

    # Save and visualize results
    save_results(results, args.output_dir)
    visualize_results(results, args.output_dir)

    print(f"\nBenchmark complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
