"""
Attack Simulator — Generates synthetic normal + attack traffic for ML trust model training.

Produces a labeled dataset of behavioral features matching the FeatureExtractor format.
Attack types simulated:
  1. Replay Attack    — reuses old queries, duplicate signatures
  2. Flood Attack     — burst of queries in very short time windows
  3. Epoch Manipulation — queries request wrong/expired epochs
  4. Sybil Attack     — rapid creation of fake identities
  5. Probing Attack   — methodical exploration with unusual payloads

Output: training_data.csv with 8 features + label (0=normal, 1=malicious)
"""

import os
import csv
import random
import math
import numpy as np
from collections import defaultdict


# Feature names (must match FeatureExtractor output)
FEATURE_NAMES = [
    "query_frequency",
    "epoch_mismatch_rate",
    "sig_failure_rate",
    "inter_query_interval",
    "payload_size_deviation",
    "time_of_day_sin",
    "time_of_day_cos",
    "query_burst_count",
]


def _random_time_encoding():
    """Generate random time-of-day encoding (sine/cosine)."""
    hour = random.uniform(0, 24)
    return (
        math.sin(2 * math.pi * hour / 24.0),
        math.cos(2 * math.pi * hour / 24.0),
    )


def generate_normal_traffic(n_samples=3000, seed=42):
    """
    Generate feature vectors for NORMAL IoT device behavior.

    Normal behavior characteristics:
    - Low query frequency (1-8 queries/min)
    - Very low epoch mismatch rate (0-5%)
    - Zero signature failure rate
    - Regular inter-query intervals (2-30 seconds)
    - Low payload size deviation
    - Occasional bursts but within reason (1-3)
    """
    rng = random.Random(seed)
    samples = []

    for _ in range(n_samples):
        tod_sin, tod_cos = _random_time_encoding()

        sample = {
            "query_frequency": rng.uniform(1, 8),
            "epoch_mismatch_rate": rng.uniform(0.0, 0.05),
            "sig_failure_rate": 0.0,  # Normal devices never fail signature
            "inter_query_interval": rng.uniform(2.0, 30.0),
            "payload_size_deviation": rng.uniform(0.0, 50.0),
            "time_of_day_sin": tod_sin + rng.gauss(0, 0.05),
            "time_of_day_cos": tod_cos + rng.gauss(0, 0.05),
            "query_burst_count": rng.choices([1, 2, 3], weights=[70, 25, 5])[0],
            "label": 0,
            "attack_type": "normal",
        }
        samples.append(sample)

    return samples


def generate_replay_attack(n_samples=300, seed=100):
    """
    Replay Attack: Reuses old valid queries.
    Features: moderate frequency, some epoch mismatches, zero sig failures
    (because replayed signatures were originally valid).
    """
    rng = random.Random(seed)
    samples = []

    for _ in range(n_samples):
        tod_sin, tod_cos = _random_time_encoding()

        sample = {
            "query_frequency": rng.uniform(5, 20),       # Moderate-high
            "epoch_mismatch_rate": rng.uniform(0.3, 0.8),  # HIGH — replaying old epoch queries
            "sig_failure_rate": rng.uniform(0.0, 0.1),     # Low — signatures were valid originally
            "inter_query_interval": rng.uniform(0.5, 5.0),  # Faster than normal
            "payload_size_deviation": rng.uniform(0.0, 20.0),  # Low — exact replays
            "time_of_day_sin": tod_sin + rng.gauss(0, 0.05),
            "time_of_day_cos": tod_cos + rng.gauss(0, 0.05),
            "query_burst_count": rng.randint(2, 8),
            "label": 1,
            "attack_type": "replay",
        }
        samples.append(sample)

    return samples


def generate_flood_attack(n_samples=300, seed=200):
    """
    Flood Attack: Overwhelm with massive query volume.
    Features: very high frequency, short intervals, large burst counts.
    """
    rng = random.Random(seed)
    samples = []

    for _ in range(n_samples):
        tod_sin, tod_cos = _random_time_encoding()

        sample = {
            "query_frequency": rng.uniform(30, 100),      # VERY HIGH
            "epoch_mismatch_rate": rng.uniform(0.0, 0.2),  # Can be low (valid epoch)
            "sig_failure_rate": rng.uniform(0.0, 0.3),     # Some may have bad sigs
            "inter_query_interval": rng.uniform(0.01, 0.5),  # VERY SHORT intervals
            "payload_size_deviation": rng.uniform(0.0, 30.0),
            "time_of_day_sin": tod_sin + rng.gauss(0, 0.05),
            "time_of_day_cos": tod_cos + rng.gauss(0, 0.05),
            "query_burst_count": rng.randint(10, 50),      # VERY HIGH burst
            "label": 1,
            "attack_type": "flood",
        }
        samples.append(sample)

    return samples


def generate_epoch_manipulation(n_samples=300, seed=300):
    """
    Epoch Manipulation: Deliberately queries wrong/future/past epochs.
    Features: very high epoch mismatch rate, otherwise somewhat normal.
    """
    rng = random.Random(seed)
    samples = []

    for _ in range(n_samples):
        tod_sin, tod_cos = _random_time_encoding()

        sample = {
            "query_frequency": rng.uniform(3, 15),
            "epoch_mismatch_rate": rng.uniform(0.5, 1.0),  # VERY HIGH mismatch
            "sig_failure_rate": rng.uniform(0.0, 0.2),
            "inter_query_interval": rng.uniform(1.0, 10.0),
            "payload_size_deviation": rng.uniform(10.0, 80.0),
            "time_of_day_sin": tod_sin + rng.gauss(0, 0.05),
            "time_of_day_cos": tod_cos + rng.gauss(0, 0.05),
            "query_burst_count": rng.randint(1, 6),
            "label": 1,
            "attack_type": "epoch_manipulation",
        }
        samples.append(sample)

    return samples


def generate_sybil_attack(n_samples=300, seed=400):
    """
    Sybil Attack: Multiple fake identities with suspicious patterns.
    Features: high frequency from rapid creation, variable payload sizes.
    """
    rng = random.Random(seed)
    samples = []

    for _ in range(n_samples):
        tod_sin, tod_cos = _random_time_encoding()

        sample = {
            "query_frequency": rng.uniform(10, 40),       # HIGH — many fake IDs querying
            "epoch_mismatch_rate": rng.uniform(0.1, 0.4),
            "sig_failure_rate": rng.uniform(0.1, 0.5),     # Some fake IDs have bad keys
            "inter_query_interval": rng.uniform(0.1, 2.0),  # SHORT — rapid creation
            "payload_size_deviation": rng.uniform(30.0, 120.0),  # HIGH variation
            "time_of_day_sin": tod_sin + rng.gauss(0, 0.1),
            "time_of_day_cos": tod_cos + rng.gauss(0, 0.1),
            "query_burst_count": rng.randint(3, 15),
            "label": 1,
            "attack_type": "sybil",
        }
        samples.append(sample)

    return samples


def generate_probing_attack(n_samples=300, seed=500):
    """
    Probing Attack: Slow, methodical exploration of the system.
    Features: low frequency but unusual payload sizes and epoch exploration.
    """
    rng = random.Random(seed)
    samples = []

    for _ in range(n_samples):
        tod_sin, tod_cos = _random_time_encoding()

        sample = {
            "query_frequency": rng.uniform(2, 10),         # Low-moderate
            "epoch_mismatch_rate": rng.uniform(0.2, 0.6),  # Exploring different epochs
            "sig_failure_rate": rng.uniform(0.0, 0.15),
            "inter_query_interval": rng.uniform(5.0, 60.0),  # Slow and deliberate
            "payload_size_deviation": rng.uniform(50.0, 200.0),  # HIGH — probing different sizes
            "time_of_day_sin": tod_sin + rng.gauss(0, 0.1),
            "time_of_day_cos": tod_cos + rng.gauss(0, 0.1),
            "query_burst_count": rng.choices([1, 2], weights=[80, 20])[0],
            "label": 1,
            "attack_type": "probing",
        }
        samples.append(sample)

    return samples


def generate_full_dataset(
    n_normal=3000,
    n_replay=300,
    n_flood=300,
    n_epoch=300,
    n_sybil=300,
    n_probing=300,
    seed=42,
):
    """
    Generate the complete labeled training dataset.

    Default: 3000 normal + 1500 attack = 4500 total samples.
    Attack ratio ~33% which gives good class balance for training.
    """
    all_samples = []
    all_samples.extend(generate_normal_traffic(n_normal, seed))
    all_samples.extend(generate_replay_attack(n_replay, seed + 100))
    all_samples.extend(generate_flood_attack(n_flood, seed + 200))
    all_samples.extend(generate_epoch_manipulation(n_epoch, seed + 300))
    all_samples.extend(generate_sybil_attack(n_sybil, seed + 400))
    all_samples.extend(generate_probing_attack(n_probing, seed + 500))

    # Shuffle
    rng = random.Random(seed)
    rng.shuffle(all_samples)

    return all_samples


def save_dataset(samples, path="training_data.csv"):
    """Save samples to CSV."""
    if not samples:
        return
    fieldnames = FEATURE_NAMES + ["label", "attack_type"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in samples:
            writer.writerow(s)
    print(f"  Saved {len(samples)} samples to {path}", flush=True)


def load_dataset(path="training_data.csv"):
    """Load dataset from CSV, returning (feature_matrix, labels, attack_types)."""
    import pandas as pd
    df = pd.read_csv(path)
    X = df[FEATURE_NAMES].values
    y = df["label"].values
    attack_types = df["attack_type"].values
    return X, y, attack_types


def print_dataset_summary(samples):
    """Print a summary of the generated dataset."""
    from collections import Counter
    type_counts = Counter(s["attack_type"] for s in samples)
    label_counts = Counter(s["label"] for s in samples)

    print("\n" + "=" * 50, flush=True)
    print("  Attack Simulator — Dataset Summary", flush=True)
    print("=" * 50, flush=True)
    print(f"  Total samples: {len(samples)}", flush=True)
    print(f"  Normal (label=0): {label_counts[0]}", flush=True)
    print(f"  Malicious (label=1): {label_counts[1]}", flush=True)
    print(f"  Attack ratio: {label_counts[1] / len(samples):.1%}", flush=True)
    print(f"\n  Attack type breakdown:", flush=True)
    for attack_type, count in sorted(type_counts.items()):
        print(f"    {attack_type:20s}: {count}", flush=True)
    print("=" * 50, flush=True)


# --------------------------------------------
# Main — Generate dataset when run directly
# --------------------------------------------

if __name__ == "__main__":
    print("Generating synthetic training dataset...", flush=True)
    samples = generate_full_dataset()
    print_dataset_summary(samples)
    save_dataset(samples, path="training_data.csv")
    print("\nDone. Dataset ready for ML training.", flush=True)
