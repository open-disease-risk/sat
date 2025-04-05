"""Benchmark ListMLE losses against other ranking losses for survival analysis."""

import os
import tempfile
import time
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from sat.loss.ranking.event_list_mle import EventListMLELoss
from sat.loss.ranking.multievent import MultiEventRankingLoss
from sat.loss.ranking.sample import SampleRankingLoss
from sat.loss.ranking.sample_list_mle import SampleListMLELoss
from sat.models.heads import SAOutput


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
        targets[i, 3 * num_events + event_type] = 1 + (
            i % 90
        )  # Set duration within range
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


def benchmark_losses(
    batch_sizes: List[int] = [8, 16, 32, 64, 128],
    num_events: int = 2,
    num_cuts: int = 10,
    num_iterations: int = 10,
):
    """Benchmark ranking losses for different batch sizes."""
    # Create files needed for loss initialization
    duration_cuts_file = create_duration_cuts_file(num_cuts)
    importance_weights_file = create_importance_weights_file(num_events)

    try:
        # Dictionary to store results
        results = {
            "batch_size": [],
            "multievent_forward": [],
            "sample_forward": [],
            "sample_list_mle_forward": [],
            "event_list_mle_forward": [],
            "multievent_backward": [],
            "sample_backward": [],
            "sample_list_mle_backward": [],
            "event_list_mle_backward": [],
        }

        # Create loss instances once to initialize any buffers
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

        event_list_mle_loss = EventListMLELoss(
            duration_cuts=duration_cuts_file,
            importance_sample_weights=importance_weights_file,
            num_events=num_events,
            epsilon=1e-10,
            temperature=1.0,
        )

        # Run benchmarks
        for batch_size in batch_sizes:
            print(f"Benchmarking batch size {batch_size}...")

            # Create new instances for each test
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

            event_list_mle_loss = EventListMLELoss(
                duration_cuts=duration_cuts_file,
                importance_sample_weights=importance_weights_file,
                num_events=num_events,
                epsilon=1e-10,
                temperature=1.0,
            )

            # Time forward pass - MultiEventRankingLoss
            multi_forward_times = []
            multi_backward_times = []

            # Time forward pass - SampleRankingLoss
            sample_forward_times = []
            sample_backward_times = []

            # Time forward pass - SampleListMLELoss
            sample_list_mle_forward_times = []
            sample_list_mle_backward_times = []

            # Time forward pass - EventListMLELoss
            event_list_mle_forward_times = []
            event_list_mle_backward_times = []

            for _ in range(num_iterations):
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

                # Clean CUDA cache if available
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # MultiEventRankingLoss - Forward pass
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start_time = time.time()
                loss = multievent_loss(predictions, targets)
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                end_time = time.time()
                multi_forward_times.append(end_time - start_time)

                # MultiEventRankingLoss - Backward pass
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start_time = time.time()
                loss.backward()
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                end_time = time.time()
                multi_backward_times.append(end_time - start_time)

                # Reset gradients
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

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

                # SampleRankingLoss - Forward pass
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start_time = time.time()
                loss = sample_loss(predictions, targets)
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                end_time = time.time()
                sample_forward_times.append(end_time - start_time)

                # SampleRankingLoss - Backward pass
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start_time = time.time()
                loss.backward()
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                end_time = time.time()
                sample_backward_times.append(end_time - start_time)

                # Reset gradients
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

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

                # SampleListMLELoss - Forward pass
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start_time = time.time()
                loss = sample_list_mle_loss(predictions, targets)
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                end_time = time.time()
                sample_list_mle_forward_times.append(end_time - start_time)

                # SampleListMLELoss - Backward pass
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start_time = time.time()
                loss.backward()
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                end_time = time.time()
                sample_list_mle_backward_times.append(end_time - start_time)

                # Reset gradients
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

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

                # EventListMLELoss - Forward pass
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start_time = time.time()
                loss = event_list_mle_loss(predictions, targets)
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                end_time = time.time()
                event_list_mle_forward_times.append(end_time - start_time)

                # EventListMLELoss - Backward pass
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start_time = time.time()
                loss.backward()
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                end_time = time.time()
                event_list_mle_backward_times.append(end_time - start_time)

            # Calculate average times (in ms)
            avg_multi_forward = np.mean(multi_forward_times) * 1000
            avg_sample_forward = np.mean(sample_forward_times) * 1000
            avg_sample_list_mle_forward = np.mean(sample_list_mle_forward_times) * 1000
            avg_event_list_mle_forward = np.mean(event_list_mle_forward_times) * 1000

            avg_multi_backward = np.mean(multi_backward_times) * 1000
            avg_sample_backward = np.mean(sample_backward_times) * 1000
            avg_sample_list_mle_backward = (
                np.mean(sample_list_mle_backward_times) * 1000
            )
            avg_event_list_mle_backward = np.mean(event_list_mle_backward_times) * 1000

            # Store results
            results["batch_size"].append(batch_size)
            results["multievent_forward"].append(avg_multi_forward)
            results["sample_forward"].append(avg_sample_forward)
            results["sample_list_mle_forward"].append(avg_sample_list_mle_forward)
            results["event_list_mle_forward"].append(avg_event_list_mle_forward)

            results["multievent_backward"].append(avg_multi_backward)
            results["sample_backward"].append(avg_sample_backward)
            results["sample_list_mle_backward"].append(avg_sample_list_mle_backward)
            results["event_list_mle_backward"].append(avg_event_list_mle_backward)

            # Print results for this batch size
            print(
                f"  MultiEventRankingLoss   - Forward: {avg_multi_forward:.4f} ms, Backward: {avg_multi_backward:.4f} ms"
            )
            print(
                f"  SampleRankingLoss       - Forward: {avg_sample_forward:.4f} ms, Backward: {avg_sample_backward:.4f} ms"
            )
            print(
                f"  SampleListMLELoss       - Forward: {avg_sample_list_mle_forward:.4f} ms, Backward: {avg_sample_list_mle_backward:.4f} ms"
            )
            print(
                f"  EventListMLELoss        - Forward: {avg_event_list_mle_forward:.4f} ms, Backward: {avg_event_list_mle_backward:.4f} ms"
            )
            print("")

        # Return results
        return results

    finally:
        # Clean up temporary files
        os.unlink(duration_cuts_file)
        os.unlink(importance_weights_file)


def benchmark_num_events(
    batch_size: int = 32,
    num_events_list: List[int] = [1, 2, 4, 8, 16],
    num_cuts: int = 10,
    num_iterations: int = 10,
):
    """Benchmark ranking losses for different numbers of events."""
    results = {
        "num_events": [],
        "multievent_forward": [],
        "sample_forward": [],
        "sample_list_mle_forward": [],
        "event_list_mle_forward": [],
        "multievent_backward": [],
        "sample_backward": [],
        "sample_list_mle_backward": [],
        "event_list_mle_backward": [],
    }

    for num_events in num_events_list:
        print(f"Benchmarking {num_events} events...")

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

            event_list_mle_loss = EventListMLELoss(
                duration_cuts=duration_cuts_file,
                importance_sample_weights=importance_weights_file,
                num_events=num_events,
                epsilon=1e-10,
                temperature=1.0,
            )

            # Time forward pass - MultiEventRankingLoss
            multi_forward_times = []
            multi_backward_times = []

            # Time forward pass - SampleRankingLoss
            sample_forward_times = []
            sample_backward_times = []

            # Time forward pass - SampleListMLELoss
            sample_list_mle_forward_times = []
            sample_list_mle_backward_times = []

            # Time forward pass - EventListMLELoss
            event_list_mle_forward_times = []
            event_list_mle_backward_times = []

            for _ in range(num_iterations):
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

                # Clean CUDA cache if available
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # MultiEventRankingLoss - Forward pass
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start_time = time.time()
                loss = multievent_loss(predictions, targets)
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                end_time = time.time()
                multi_forward_times.append(end_time - start_time)

                # MultiEventRankingLoss - Backward pass
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start_time = time.time()
                loss.backward()
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                end_time = time.time()
                multi_backward_times.append(end_time - start_time)

                # Reset gradients
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

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

                # SampleRankingLoss - Forward pass
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start_time = time.time()
                loss = sample_loss(predictions, targets)
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                end_time = time.time()
                sample_forward_times.append(end_time - start_time)

                # SampleRankingLoss - Backward pass
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start_time = time.time()
                loss.backward()
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                end_time = time.time()
                sample_backward_times.append(end_time - start_time)

                # Reset gradients
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

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

                # SampleListMLELoss - Forward pass
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start_time = time.time()
                loss = sample_list_mle_loss(predictions, targets)
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                end_time = time.time()
                sample_list_mle_forward_times.append(end_time - start_time)

                # SampleListMLELoss - Backward pass
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start_time = time.time()
                loss.backward()
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                end_time = time.time()
                sample_list_mle_backward_times.append(end_time - start_time)

                # Reset gradients
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

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

                # EventListMLELoss - Forward pass
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start_time = time.time()
                loss = event_list_mle_loss(predictions, targets)
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                end_time = time.time()
                event_list_mle_forward_times.append(end_time - start_time)

                # EventListMLELoss - Backward pass
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start_time = time.time()
                loss.backward()
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                end_time = time.time()
                event_list_mle_backward_times.append(end_time - start_time)

            # Calculate average times (in ms)
            avg_multi_forward = np.mean(multi_forward_times) * 1000
            avg_sample_forward = np.mean(sample_forward_times) * 1000
            avg_sample_list_mle_forward = np.mean(sample_list_mle_forward_times) * 1000
            avg_event_list_mle_forward = np.mean(event_list_mle_forward_times) * 1000

            avg_multi_backward = np.mean(multi_backward_times) * 1000
            avg_sample_backward = np.mean(sample_backward_times) * 1000
            avg_sample_list_mle_backward = (
                np.mean(sample_list_mle_backward_times) * 1000
            )
            avg_event_list_mle_backward = np.mean(event_list_mle_backward_times) * 1000

            # Store results
            results["num_events"].append(num_events)
            results["multievent_forward"].append(avg_multi_forward)
            results["sample_forward"].append(avg_sample_forward)
            results["sample_list_mle_forward"].append(avg_sample_list_mle_forward)
            results["event_list_mle_forward"].append(avg_event_list_mle_forward)

            results["multievent_backward"].append(avg_multi_backward)
            results["sample_backward"].append(avg_sample_backward)
            results["sample_list_mle_backward"].append(avg_sample_list_mle_backward)
            results["event_list_mle_backward"].append(avg_event_list_mle_backward)

            # Print results for this number of events
            print(
                f"  MultiEventRankingLoss   - Forward: {avg_multi_forward:.4f} ms, Backward: {avg_multi_backward:.4f} ms"
            )
            print(
                f"  SampleRankingLoss       - Forward: {avg_sample_forward:.4f} ms, Backward: {avg_sample_backward:.4f} ms"
            )
            print(
                f"  SampleListMLELoss       - Forward: {avg_sample_list_mle_forward:.4f} ms, Backward: {avg_sample_list_mle_backward:.4f} ms"
            )
            print(
                f"  EventListMLELoss        - Forward: {avg_event_list_mle_forward:.4f} ms, Backward: {avg_event_list_mle_backward:.4f} ms"
            )
            print("")

        finally:
            # Clean up temporary files
            os.unlink(duration_cuts_file)
            os.unlink(importance_weights_file)

    # Return results
    return results


def plot_benchmark_results(results, title, x_label, y_label, save_path=None):
    """Plot benchmark results."""
    plt.figure(figsize=(12, 8))

    # Extract data
    x = results[x_label]

    # Plot forward pass times
    plt.plot(
        x, results["multievent_forward"], "b-o", label="MultiEventRankingLoss (Forward)"
    )
    plt.plot(x, results["sample_forward"], "r-o", label="SampleRankingLoss (Forward)")
    plt.plot(
        x,
        results["sample_list_mle_forward"],
        "g-o",
        label="SampleListMLELoss (Forward)",
    )
    plt.plot(
        x, results["event_list_mle_forward"], "m-o", label="EventListMLELoss (Forward)"
    )

    # Plot backward pass times
    plt.plot(
        x,
        results["multievent_backward"],
        "b--^",
        label="MultiEventRankingLoss (Backward)",
    )
    plt.plot(
        x, results["sample_backward"], "r--^", label="SampleRankingLoss (Backward)"
    )
    plt.plot(
        x,
        results["sample_list_mle_backward"],
        "g--^",
        label="SampleListMLELoss (Backward)",
    )
    plt.plot(
        x,
        results["event_list_mle_backward"],
        "m--^",
        label="EventListMLELoss (Backward)",
    )

    # Add labels and legend
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()

    # Save figure if path provided
    if save_path:
        plt.savefig(save_path)

    plt.show()


def save_results_to_csv(results, filename):
    """Save benchmark results to CSV file."""
    # Create logs directory if it doesn't exist
    log_dir = os.path.join(os.getcwd(), "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Prepare data for CSV
    data = []

    if "batch_size" in results:
        # Results by batch size
        for i, batch_size in enumerate(results["batch_size"]):
            row = {
                "batch_size": batch_size,
                "num_events": "N/A",
                "method": "MultiEventRankingLoss",
                "forward_time_ms": results["multievent_forward"][i],
                "backward_time_ms": results["multievent_backward"][i],
                "total_time_ms": results["multievent_forward"][i]
                + results["multievent_backward"][i],
            }
            data.append(row)

            row = {
                "batch_size": batch_size,
                "num_events": "N/A",
                "method": "SampleRankingLoss",
                "forward_time_ms": results["sample_forward"][i],
                "backward_time_ms": results["sample_backward"][i],
                "total_time_ms": results["sample_forward"][i]
                + results["sample_backward"][i],
            }
            data.append(row)

            row = {
                "batch_size": batch_size,
                "num_events": "N/A",
                "method": "SampleListMLELoss",
                "forward_time_ms": results["sample_list_mle_forward"][i],
                "backward_time_ms": results["sample_list_mle_backward"][i],
                "total_time_ms": results["sample_list_mle_forward"][i]
                + results["sample_list_mle_backward"][i],
            }
            data.append(row)

            row = {
                "batch_size": batch_size,
                "num_events": "N/A",
                "method": "EventListMLELoss",
                "forward_time_ms": results["event_list_mle_forward"][i],
                "backward_time_ms": results["event_list_mle_backward"][i],
                "total_time_ms": results["event_list_mle_forward"][i]
                + results["event_list_mle_backward"][i],
            }
            data.append(row)

    elif "num_events" in results:
        # Results by number of events
        for i, num_events in enumerate(results["num_events"]):
            row = {
                "batch_size": "N/A",
                "num_events": num_events,
                "method": "MultiEventRankingLoss",
                "forward_time_ms": results["multievent_forward"][i],
                "backward_time_ms": results["multievent_backward"][i],
                "total_time_ms": results["multievent_forward"][i]
                + results["multievent_backward"][i],
            }
            data.append(row)

            row = {
                "batch_size": "N/A",
                "num_events": num_events,
                "method": "SampleRankingLoss",
                "forward_time_ms": results["sample_forward"][i],
                "backward_time_ms": results["sample_backward"][i],
                "total_time_ms": results["sample_forward"][i]
                + results["sample_backward"][i],
            }
            data.append(row)

            row = {
                "batch_size": "N/A",
                "num_events": num_events,
                "method": "SampleListMLELoss",
                "forward_time_ms": results["sample_list_mle_forward"][i],
                "backward_time_ms": results["sample_list_mle_backward"][i],
                "total_time_ms": results["sample_list_mle_forward"][i]
                + results["sample_list_mle_backward"][i],
            }
            data.append(row)

            row = {
                "batch_size": "N/A",
                "num_events": num_events,
                "method": "EventListMLELoss",
                "forward_time_ms": results["event_list_mle_forward"][i],
                "backward_time_ms": results["event_list_mle_backward"][i],
                "total_time_ms": results["event_list_mle_forward"][i]
                + results["event_list_mle_backward"][i],
            }
            data.append(row)

    # Create DataFrame and save to CSV
    df = pd.DataFrame(data)
    csv_path = os.path.join(log_dir, filename)
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")

    return csv_path


def generate_summary_markdown(batch_results_path, event_results_path):
    """Generate a summary markdown file with key findings."""
    # Read CSV files
    batch_df = pd.read_csv(batch_results_path)
    event_df = pd.read_csv(event_results_path)

    # Calculate speedups and comparisons
    # Reference is SampleRankingLoss (the original implementation)

    # For batch size comparison
    batch_summary = {}
    for batch_size in batch_df["batch_size"].unique():
        if batch_size == "N/A":
            continue

        batch_subset = batch_df[batch_df["batch_size"] == batch_size]
        reference = batch_subset[batch_subset["method"] == "SampleRankingLoss"][
            "total_time_ms"
        ].values[0]

        batch_summary[batch_size] = {}
        for method in batch_subset["method"].unique():
            if method == "SampleRankingLoss":
                continue

            method_time = batch_subset[batch_subset["method"] == method][
                "total_time_ms"
            ].values[0]
            speedup = reference / method_time
            batch_summary[batch_size][method] = speedup

    # For event count comparison
    event_summary = {}
    for num_events in event_df["num_events"].unique():
        if num_events == "N/A":
            continue

        event_subset = event_df[event_df["num_events"] == num_events]
        reference = event_subset[event_subset["method"] == "SampleRankingLoss"][
            "total_time_ms"
        ].values[0]

        event_summary[num_events] = {}
        for method in event_subset["method"].unique():
            if method == "SampleRankingLoss":
                continue

            method_time = event_subset[event_subset["method"] == method][
                "total_time_ms"
            ].values[0]
            speedup = reference / method_time
            event_summary[num_events][method] = speedup

    # Generate markdown content
    markdown = "# Ranking Loss Benchmark Summary\n\n"
    markdown += "This document summarizes the performance comparison between different ranking loss implementations.\n\n"

    # Batch size comparison section
    markdown += "## Batch Size Scaling\n\n"
    markdown += "The table below shows the speedup factor compared to SampleRankingLoss (higher is better).\n\n"

    # Create batch size comparison table
    markdown += "| Batch Size | MultiEventRankingLoss | SampleListMLELoss | EventListMLELoss |\n"
    markdown += (
        "|------------|----------------------|-------------------|------------------|\n"
    )

    for batch_size in sorted(batch_summary.keys()):
        summary = batch_summary[batch_size]
        markdown += f"| {batch_size} | {summary.get('MultiEventRankingLoss', 'N/A'):.2f}x | {summary.get('SampleListMLELoss', 'N/A'):.2f}x | {summary.get('EventListMLELoss', 'N/A'):.2f}x |\n"

    # Event count comparison section
    markdown += "\n## Event Count Scaling\n\n"
    markdown += "The table below shows the speedup factor compared to SampleRankingLoss (higher is better).\n\n"

    # Create event count comparison table
    markdown += "| Event Count | MultiEventRankingLoss | SampleListMLELoss | EventListMLELoss |\n"
    markdown += (
        "|------------|----------------------|-------------------|------------------|\n"
    )

    for num_events in sorted(event_summary.keys()):
        summary = event_summary[num_events]
        markdown += f"| {num_events} | {summary.get('MultiEventRankingLoss', 'N/A'):.2f}x | {summary.get('SampleListMLELoss', 'N/A'):.2f}x | {summary.get('EventListMLELoss', 'N/A'):.2f}x |\n"

    # Overall findings
    markdown += "\n## Key Findings\n\n"

    # Calculate average speedups across all configurations
    all_multi = []
    all_sample_list = []
    all_event_list = []

    for batch_size in batch_summary:
        all_multi.append(batch_summary[batch_size].get("MultiEventRankingLoss", 0))
        all_sample_list.append(batch_summary[batch_size].get("SampleListMLELoss", 0))
        all_event_list.append(batch_summary[batch_size].get("EventListMLELoss", 0))

    for num_events in event_summary:
        all_multi.append(event_summary[num_events].get("MultiEventRankingLoss", 0))
        all_sample_list.append(event_summary[num_events].get("SampleListMLELoss", 0))
        all_event_list.append(event_summary[num_events].get("EventListMLELoss", 0))

    avg_multi = np.mean(all_multi) if all_multi else 0
    avg_sample_list = np.mean(all_sample_list) if all_sample_list else 0
    avg_event_list = np.mean(all_event_list) if all_event_list else 0

    markdown += f"- **MultiEventRankingLoss**: Average {avg_multi:.2f}x speedup compared to SampleRankingLoss\n"
    markdown += f"- **SampleListMLELoss**: Average {avg_sample_list:.2f}x speedup compared to SampleRankingLoss\n"
    markdown += f"- **EventListMLELoss**: Average {avg_event_list:.2f}x speedup compared to SampleRankingLoss\n\n"

    # Best performing method
    best_method = (
        "MultiEventRankingLoss"
        if avg_multi > max(avg_sample_list, avg_event_list)
        else (
            "SampleListMLELoss"
            if avg_sample_list > avg_event_list
            else "EventListMLELoss"
        )
    )

    markdown += f"**Overall Best Performer**: {best_method}\n\n"

    # Add scaling observations
    markdown += "### Scaling Observations\n\n"

    # Check batch size scaling pattern for each method
    multi_batch_scaling = []
    sample_list_batch_scaling = []
    event_list_batch_scaling = []

    batch_sizes = sorted(batch_summary.keys())
    for i in range(1, len(batch_sizes)):
        if (
            "MultiEventRankingLoss" in batch_summary[batch_sizes[i]]
            and "MultiEventRankingLoss" in batch_summary[batch_sizes[i - 1]]
        ):
            ratio = (
                batch_summary[batch_sizes[i]]["MultiEventRankingLoss"]
                / batch_summary[batch_sizes[i - 1]]["MultiEventRankingLoss"]
            )
            multi_batch_scaling.append(ratio)

        if (
            "SampleListMLELoss" in batch_summary[batch_sizes[i]]
            and "SampleListMLELoss" in batch_summary[batch_sizes[i - 1]]
        ):
            ratio = (
                batch_summary[batch_sizes[i]]["SampleListMLELoss"]
                / batch_summary[batch_sizes[i - 1]]["SampleListMLELoss"]
            )
            sample_list_batch_scaling.append(ratio)

        if (
            "EventListMLELoss" in batch_summary[batch_sizes[i]]
            and "EventListMLELoss" in batch_summary[batch_sizes[i - 1]]
        ):
            ratio = (
                batch_summary[batch_sizes[i]]["EventListMLELoss"]
                / batch_summary[batch_sizes[i - 1]]["EventListMLELoss"]
            )
            event_list_batch_scaling.append(ratio)

    # Determine if the method scales better or worse with batch size
    if multi_batch_scaling and np.mean(multi_batch_scaling) > 1:
        markdown += "- MultiEventRankingLoss scales better with larger batch sizes\n"
    elif multi_batch_scaling:
        markdown += "- MultiEventRankingLoss scales worse with larger batch sizes\n"

    if sample_list_batch_scaling and np.mean(sample_list_batch_scaling) > 1:
        markdown += "- SampleListMLELoss scales better with larger batch sizes\n"
    elif sample_list_batch_scaling:
        markdown += "- SampleListMLELoss scales worse with larger batch sizes\n"

    if event_list_batch_scaling and np.mean(event_list_batch_scaling) > 1:
        markdown += "- EventListMLELoss scales better with larger batch sizes\n"
    elif event_list_batch_scaling:
        markdown += "- EventListMLELoss scales worse with larger batch sizes\n"

    # Save markdown to file
    log_dir = os.path.join(os.getcwd(), "logs")
    md_path = os.path.join(log_dir, "ranking_loss_comparison.md")

    with open(md_path, "w") as f:
        f.write(markdown)

    print(f"Summary saved to {md_path}")
    return md_path


def main():
    """Run benchmarks and plot results."""
    print("Running batch size benchmarks...")
    batch_results = benchmark_losses(
        batch_sizes=[16, 32], num_events=2, num_cuts=10, num_iterations=3
    )

    print("\nRunning num_events benchmarks...")
    events_results = benchmark_num_events(
        batch_size=32, num_events_list=[1, 2], num_cuts=10, num_iterations=3
    )

    # Save results to CSV
    batch_csv = save_results_to_csv(batch_results, "list_mle_batch_benchmark.csv")
    events_csv = save_results_to_csv(events_results, "list_mle_events_benchmark.csv")

    # Generate summary markdown
    summary_md = generate_summary_markdown(batch_csv, events_csv)

    # Check if matplotlib is available for visualization
    matplotlib_available = True
    try:
        import matplotlib

        matplotlib.use("Agg")  # Use non-interactive backend
    except ImportError:
        matplotlib_available = False
        print("Matplotlib not available - skipping visualization")

    # Plot results if matplotlib is available
    if matplotlib_available:
        # Plot batch size results
        plot_benchmark_results(
            batch_results,
            "Performance Comparison by Batch Size",
            "batch_size",
            "Time (ms)",
            save_path="logs/list_mle_batch_comparison.png",
        )

        # Plot event count results
        plot_benchmark_results(
            events_results,
            "Performance Comparison by Number of Events",
            "num_events",
            "Time (ms)",
            save_path="logs/list_mle_events_comparison.png",
        )


if __name__ == "__main__":
    main()
