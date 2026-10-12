"""
Attack Simulator — Generates synthetic normal + attack traffic for ML trust model training.

Produces a labeled dataset of behavioral features matching the FeatureExtractor format.
Attack types simulated:
  1. Replay Attack    — reuses old queries, duplicate signatures
  2. Flood Attack     — burst of queries in very short time windows
  3. Epoch Manipulation — queries request wrong/expired epochs
  4. Sybil Attack     — rapid creation of fake identities
  5. Probing Attack   — methodical exploration with unusual payloads

Noise injection strategies (v2 — realistic):
  - Feature jitter: Gaussian noise on every feature to simulate sensor imprecision
  - Borderline samples: Normal traffic that looks suspicious, attacks that look benign
  - Label noise: Small fraction of deliberately mislabeled samples (~2-3%)
  - Overlapping distributions: Ranges intentionally overlap between classes
  - Outlier injection: Rare extreme values in normal traffic

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


def _add_feature_jitter(sample, rng, noise_scale=0.10):
    """
    Add Gaussian jitter to all numeric features to simulate real-world
    sensor noise, measurement imprecision, and network timing variance.
    """
    noisy = {}
    for key, val in sample.items():
        if key in ("label", "attack_type"):
            noisy[key] = val
            continue
        if isinstance(val, (int, float)):
            noise = rng.gauss(0, abs(val) * noise_scale + 0.01)
            noisy_val = val + noise
            # Clamp rates to [0, 1]
            if "rate" in key:
                noisy_val = max(0.0, min(1.0, noisy_val))
            # Clamp non-negative features
            elif key in ("query_frequency", "inter_query_interval",
                         "payload_size_deviation", "query_burst_count"):
                noisy_val = max(0.0, noisy_val)
            # Round burst count to integer
            if key == "query_burst_count":
                noisy_val = max(1, round(noisy_val))
            noisy[key] = noisy_val
        else:
            noisy[key] = val
    return noisy


def generate_normal_traffic(n_samples=3000, seed=42):
    """
    Generate feature vectors for NORMAL IoT device behavior.

    v2: Added realistic noise and edge cases:
    - Some normal devices occasionally show elevated query rates (busy periods)
    - Rare but real sig_failure_rate > 0 (network glitches, firmware bugs)
    - Occasional epoch mismatches from clock drift
    - Wide variance in inter-query intervals
    - Outlier injection (~5% of samples behave unusually)
    """
    rng = random.Random(seed)
    samples = []

    for i in range(n_samples):
        tod_sin, tod_cos = _random_time_encoding()

        # ~5% of normal traffic is "edge-case" — looks suspicious but isn't
        is_edge_case = rng.random() < 0.05

        if is_edge_case:
            # Edge-case normal: busy device, occasional hiccups
            sample = {
                "query_frequency": rng.uniform(6, 18),           # Higher than typical
                "epoch_mismatch_rate": rng.uniform(0.0, 0.15),   # Some clock drift
                "sig_failure_rate": rng.uniform(0.0, 0.08),      # Rare firmware glitches
                "inter_query_interval": rng.uniform(0.5, 8.0),   # Faster during busy periods
                "payload_size_deviation": rng.uniform(20.0, 90.0),  # More variation
                "time_of_day_sin": tod_sin + rng.gauss(0, 0.12),
                "time_of_day_cos": tod_cos + rng.gauss(0, 0.12),
                "query_burst_count": rng.choices([2, 3, 4, 5], weights=[40, 30, 20, 10])[0],
                "label": 0,
                "attack_type": "normal",
            }
        else:
            # Standard normal behavior with wider variance than v1
            sample = {
                "query_frequency": rng.uniform(0.5, 12),         # Wider range
                "epoch_mismatch_rate": rng.uniform(0.0, 0.08),   # Slight clock drift possible
                "sig_failure_rate": rng.choices(
                    [0.0, rng.uniform(0.001, 0.03)],
                    weights=[92, 8]
                )[0],  # Rarely non-zero
                "inter_query_interval": rng.uniform(1.5, 45.0),  # Wider range
                "payload_size_deviation": rng.uniform(0.0, 60.0),
                "time_of_day_sin": tod_sin + rng.gauss(0, 0.08),
                "time_of_day_cos": tod_cos + rng.gauss(0, 0.08),
                "query_burst_count": rng.choices([1, 2, 3], weights=[60, 30, 10])[0],
                "label": 0,
                "attack_type": "normal",
            }

        # Apply feature jitter
        sample = _add_feature_jitter(sample, rng, noise_scale=0.08)
        samples.append(sample)

    return samples


def generate_replay_attack(n_samples=300, seed=100):
    """
    Replay Attack: Reuses old valid queries.

    v2: Some replay attacks are stealthy (low rate, mimics normal),
    others are aggressive. Added variance overlap with normal traffic.
    """
    rng = random.Random(seed)
    samples = []

    for _ in range(n_samples):
        tod_sin, tod_cos = _random_time_encoding()

        # 20% are "stealthy" replays that look more normal
        is_stealthy = rng.random() < 0.20

        if is_stealthy:
            sample = {
                "query_frequency": rng.uniform(3, 12),
                "epoch_mismatch_rate": rng.uniform(0.10, 0.35),   # Lower than obvious
                "sig_failure_rate": rng.uniform(0.0, 0.05),
                "inter_query_interval": rng.uniform(2.0, 15.0),   # Mimics normal pacing
                "payload_size_deviation": rng.uniform(0.0, 30.0),
                "time_of_day_sin": tod_sin + rng.gauss(0, 0.08),
                "time_of_day_cos": tod_cos + rng.gauss(0, 0.08),
                "query_burst_count": rng.randint(1, 4),
                "label": 1,
                "attack_type": "replay",
            }
        else:
            sample = {
                "query_frequency": rng.uniform(5, 25),
                "epoch_mismatch_rate": rng.uniform(0.25, 0.85),
                "sig_failure_rate": rng.uniform(0.0, 0.12),
                "inter_query_interval": rng.uniform(0.3, 6.0),
                "payload_size_deviation": rng.uniform(0.0, 25.0),
                "time_of_day_sin": tod_sin + rng.gauss(0, 0.08),
                "time_of_day_cos": tod_cos + rng.gauss(0, 0.08),
                "query_burst_count": rng.randint(2, 10),
                "label": 1,
                "attack_type": "replay",
            }

        sample = _add_feature_jitter(sample, rng, noise_scale=0.12)
        samples.append(sample)

    return samples


def generate_flood_attack(n_samples=300, seed=200):
    """
    Flood Attack: Overwhelm with massive query volume.

    v2: Some flood attacks ramp slowly, not all are extreme bursts.
    Added moderate floods that overlap with busy normal traffic.
    """
    rng = random.Random(seed)
    samples = []

    for _ in range(n_samples):
        tod_sin, tod_cos = _random_time_encoding()

        # 15% are "slow floods" — harder to detect
        is_slow = rng.random() < 0.15

        if is_slow:
            sample = {
                "query_frequency": rng.uniform(12, 35),            # Moderate, overlaps with busy normal
                "epoch_mismatch_rate": rng.uniform(0.0, 0.15),
                "sig_failure_rate": rng.uniform(0.0, 0.15),
                "inter_query_interval": rng.uniform(0.3, 2.0),
                "payload_size_deviation": rng.uniform(0.0, 40.0),
                "time_of_day_sin": tod_sin + rng.gauss(0, 0.08),
                "time_of_day_cos": tod_cos + rng.gauss(0, 0.08),
                "query_burst_count": rng.randint(5, 20),
                "label": 1,
                "attack_type": "flood",
            }
        else:
            sample = {
                "query_frequency": rng.uniform(25, 100),
                "epoch_mismatch_rate": rng.uniform(0.0, 0.25),
                "sig_failure_rate": rng.uniform(0.0, 0.35),
                "inter_query_interval": rng.uniform(0.01, 0.8),
                "payload_size_deviation": rng.uniform(0.0, 35.0),
                "time_of_day_sin": tod_sin + rng.gauss(0, 0.08),
                "time_of_day_cos": tod_cos + rng.gauss(0, 0.08),
                "query_burst_count": rng.randint(8, 50),
                "label": 1,
                "attack_type": "flood",
            }

        sample = _add_feature_jitter(sample, rng, noise_scale=0.10)
        samples.append(sample)

    return samples


def generate_epoch_manipulation(n_samples=300, seed=300):
    """
    Epoch Manipulation: Deliberately queries wrong/future/past epochs.

    v2: Some manipulations are subtle (small mismatches), others extreme.
    Added overlap zone with normal clock-drift mismatches.
    """
    rng = random.Random(seed)
    samples = []

    for _ in range(n_samples):
        tod_sin, tod_cos = _random_time_encoding()

        # 25% are subtle manipulations (harder to distinguish from clock drift)
        is_subtle = rng.random() < 0.25

        if is_subtle:
            sample = {
                "query_frequency": rng.uniform(2, 10),
                "epoch_mismatch_rate": rng.uniform(0.12, 0.40),   # Overlaps with noisy normal
                "sig_failure_rate": rng.uniform(0.0, 0.10),
                "inter_query_interval": rng.uniform(2.0, 20.0),
                "payload_size_deviation": rng.uniform(5.0, 50.0),
                "time_of_day_sin": tod_sin + rng.gauss(0, 0.08),
                "time_of_day_cos": tod_cos + rng.gauss(0, 0.08),
                "query_burst_count": rng.randint(1, 4),
                "label": 1,
                "attack_type": "epoch_manipulation",
            }
        else:
            sample = {
                "query_frequency": rng.uniform(3, 18),
                "epoch_mismatch_rate": rng.uniform(0.40, 1.0),
                "sig_failure_rate": rng.uniform(0.0, 0.25),
                "inter_query_interval": rng.uniform(0.8, 12.0),
                "payload_size_deviation": rng.uniform(8.0, 90.0),
                "time_of_day_sin": tod_sin + rng.gauss(0, 0.10),
                "time_of_day_cos": tod_cos + rng.gauss(0, 0.10),
                "query_burst_count": rng.randint(1, 7),
                "label": 1,
                "attack_type": "epoch_manipulation",
            }

        sample = _add_feature_jitter(sample, rng, noise_scale=0.10)
        samples.append(sample)

    return samples


def generate_sybil_attack(n_samples=300, seed=400):
    """
    Sybil Attack: Multiple fake identities with suspicious patterns.

    v2: Some sybil attacks use fewer identities and blend better.
    Added variance and overlap with normal multi-device scenarios.
    """
    rng = random.Random(seed)
    samples = []

    for _ in range(n_samples):
        tod_sin, tod_cos = _random_time_encoding()

        # 20% are low-intensity sybils
        is_low_intensity = rng.random() < 0.20

        if is_low_intensity:
            sample = {
                "query_frequency": rng.uniform(6, 18),
                "epoch_mismatch_rate": rng.uniform(0.05, 0.20),
                "sig_failure_rate": rng.uniform(0.05, 0.20),
                "inter_query_interval": rng.uniform(1.0, 8.0),
                "payload_size_deviation": rng.uniform(15.0, 70.0),
                "time_of_day_sin": tod_sin + rng.gauss(0, 0.12),
                "time_of_day_cos": tod_cos + rng.gauss(0, 0.12),
                "query_burst_count": rng.randint(2, 8),
                "label": 1,
                "attack_type": "sybil",
            }
        else:
            sample = {
                "query_frequency": rng.uniform(10, 45),
                "epoch_mismatch_rate": rng.uniform(0.08, 0.45),
                "sig_failure_rate": rng.uniform(0.08, 0.55),
                "inter_query_interval": rng.uniform(0.1, 3.0),
                "payload_size_deviation": rng.uniform(25.0, 130.0),
                "time_of_day_sin": tod_sin + rng.gauss(0, 0.15),
                "time_of_day_cos": tod_cos + rng.gauss(0, 0.15),
                "query_burst_count": rng.randint(3, 18),
                "label": 1,
                "attack_type": "sybil",
            }

        sample = _add_feature_jitter(sample, rng, noise_scale=0.12)
        samples.append(sample)

    return samples


def generate_probing_attack(n_samples=300, seed=500):
    """
    Probing Attack: Slow, methodical exploration of the system.

    v2: Some probing attacks are very stealthy with near-normal features.
    Added overlap with normal exploratory behavior.
    """
    rng = random.Random(seed)
    samples = []

    for _ in range(n_samples):
        tod_sin, tod_cos = _random_time_encoding()

        # 30% are very stealthy probes
        is_stealthy = rng.random() < 0.30

        if is_stealthy:
            sample = {
                "query_frequency": rng.uniform(1, 8),              # Mimics normal
                "epoch_mismatch_rate": rng.uniform(0.05, 0.25),    # Subtle exploration
                "sig_failure_rate": rng.uniform(0.0, 0.05),
                "inter_query_interval": rng.uniform(8.0, 50.0),   # Slow and patient
                "payload_size_deviation": rng.uniform(30.0, 100.0),
                "time_of_day_sin": tod_sin + rng.gauss(0, 0.08),
                "time_of_day_cos": tod_cos + rng.gauss(0, 0.08),
                "query_burst_count": rng.choices([1, 2], weights=[70, 30])[0],
                "label": 1,
                "attack_type": "probing",
            }
        else:
            sample = {
                "query_frequency": rng.uniform(2, 12),
                "epoch_mismatch_rate": rng.uniform(0.15, 0.65),
                "sig_failure_rate": rng.uniform(0.0, 0.18),
                "inter_query_interval": rng.uniform(4.0, 65.0),
                "payload_size_deviation": rng.uniform(40.0, 220.0),
                "time_of_day_sin": tod_sin + rng.gauss(0, 0.12),
                "time_of_day_cos": tod_cos + rng.gauss(0, 0.12),
                "query_burst_count": rng.choices([1, 2, 3], weights=[60, 30, 10])[0],
                "label": 1,
                "attack_type": "probing",
            }

        sample = _add_feature_jitter(sample, rng, noise_scale=0.10)
        samples.append(sample)

    return samples


def _inject_label_noise(samples, noise_rate=0.025, seed=999):
    """
    Inject label noise: flip ~2.5% of labels to simulate
    real-world labeling errors and ambiguous traffic.
    This prevents the model from achieving suspiciously perfect accuracy.
    """
    rng = random.Random(seed)
    n_flipped = 0
    for sample in samples:
        if rng.random() < noise_rate:
            sample["label"] = 1 - sample["label"]  # Flip 0↔1
            sample["attack_type"] = "noisy_" + sample["attack_type"]
            n_flipped += 1
    return samples, n_flipped


def _inject_ambiguous_samples(n_samples=150, seed=777):
    """
    Generate borderline/ambiguous samples that sit right on the
    decision boundary. These are genuinely hard to classify.
    """
    rng = random.Random(seed)
    samples = []

    for _ in range(n_samples):
        tod_sin, tod_cos = _random_time_encoding()

        # These samples have features that are between normal and attack ranges
        sample = {
            "query_frequency": rng.uniform(6, 18),           # Overlap zone
            "epoch_mismatch_rate": rng.uniform(0.05, 0.20),  # Ambiguous range
            "sig_failure_rate": rng.uniform(0.0, 0.10),      # Could be either
            "inter_query_interval": rng.uniform(1.0, 8.0),   # Middle ground
            "payload_size_deviation": rng.uniform(20.0, 80.0),
            "time_of_day_sin": tod_sin + rng.gauss(0, 0.15),
            "time_of_day_cos": tod_cos + rng.gauss(0, 0.15),
            "query_burst_count": rng.randint(2, 6),
            # Randomly assign label — these are genuinely ambiguous
            "label": rng.choices([0, 1], weights=[55, 45])[0],
            "attack_type": "ambiguous",
        }

        sample = _add_feature_jitter(sample, rng, noise_scale=0.15)
        samples.append(sample)

    return samples


def generate_full_dataset(
    n_normal=3000,
    n_replay=300,
    n_flood=300,
    n_epoch=300,
    n_sybil=300,
    n_probing=300,
    n_ambiguous=150,
    label_noise_rate=0.025,
    seed=42,
):
    """
    Generate the complete labeled training dataset (v2 — noisy).

    Default: 3000 normal + 1500 attack + 150 ambiguous = 4650 total samples.
    Then ~2.5% label noise is injected on top.
    """
    all_samples = []
    all_samples.extend(generate_normal_traffic(n_normal, seed))
    all_samples.extend(generate_replay_attack(n_replay, seed + 100))
    all_samples.extend(generate_flood_attack(n_flood, seed + 200))
    all_samples.extend(generate_epoch_manipulation(n_epoch, seed + 300))
    all_samples.extend(generate_sybil_attack(n_sybil, seed + 400))
    all_samples.extend(generate_probing_attack(n_probing, seed + 500))

    # Add ambiguous borderline samples
    all_samples.extend(_inject_ambiguous_samples(n_ambiguous, seed + 777))

    # Inject label noise
    all_samples, n_flipped = _inject_label_noise(
        all_samples, noise_rate=label_noise_rate, seed=seed + 999
    )

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
    print("  Attack Simulator — Dataset Summary (v2 Noisy)", flush=True)
    print("=" * 50, flush=True)
    print(f"  Total samples: {len(samples)}", flush=True)
    print(f"  Normal (label=0): {label_counts[0]}", flush=True)
    print(f"  Malicious (label=1): {label_counts[1]}", flush=True)
    print(f"  Attack ratio: {label_counts[1] / len(samples):.1%}", flush=True)
    print(f"\n  Attack type breakdown:", flush=True)
    for attack_type, count in sorted(type_counts.items()):
        print(f"    {attack_type:20s}: {count}", flush=True)

    # Count noisy labels
    noisy_count = sum(1 for s in samples if s["attack_type"].startswith("noisy_"))
    print(f"\n  Label noise injected: {noisy_count} samples ({noisy_count/len(samples):.1%})", flush=True)

    # Count ambiguous
    ambig_count = sum(1 for s in samples if "ambiguous" in s["attack_type"])
    print(f"  Ambiguous borderline: {ambig_count} samples ({ambig_count/len(samples):.1%})", flush=True)
    print("=" * 50, flush=True)


# --------------------------------------------
# Main — Generate dataset when run directly
# --------------------------------------------

if __name__ == "__main__":
    print("Generating synthetic training dataset (v2 — noisy)...", flush=True)
    samples = generate_full_dataset()
    print_dataset_summary(samples)
    save_dataset(samples, path="training_data.csv")
    print("\nDone. Noisy dataset ready for ML training.", flush=True)
