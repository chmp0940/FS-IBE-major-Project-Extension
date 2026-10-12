"""
ML Trust Simulation — Benchmarks ML-enhanced trust scoring vs static scoring.
Integrates with the existing simulation pipeline.
"""
import sys, time, os, csv, json
import numpy as np

import lattice_infrastructure
sys.modules["LatticeCrypto"] = lattice_infrastructure

import lattice_infrastructure as P1
from forward_security import UserOps
from Trust_Model import (TrustManager, DilithiumStub, Query, QueryValidator,
                         match_query_to_data, FeatureExtractor)
from ml_trust_engine import MLTrustEngine
from attack_simulator import generate_full_dataset, save_dataset, FEATURE_NAMES


def _setup_system(n=64, tree_depth=3):
    """Setup crypto system and return components."""
    params = P1.LatticeParams(n=n)
    system = P1.Setup(tree_depth=tree_depth, params=params)
    ops = UserOps(system)
    sig = DilithiumStub()
    sk_user = b"user_sk_for_ml_simulation_test!!"
    pk_user = sig.pk_from_sk(sk_user)
    return params, system, ops, sig, sk_user, pk_user


def _make_query(ops, validator, sig, sk_user, user_id, epoch):
    """Generate a valid signed query."""
    keyword_ct = ops.Encrypt(user_id, epoch, 1)
    ekw = keyword_ct["c1"].tobytes()
    q = Query(encrypted_keyword=ekw, signature=b"", epoch=epoch)
    msg = validator.serialize(user_id, q)
    q.signature = sig.sign(msg, sk_user)
    return q, keyword_ct


def _make_malicious_query(ops, validator, sig, user_id, epoch, attack_type="bad_sig"):
    """Generate a malicious query."""
    keyword_ct = ops.Encrypt(user_id, epoch, 1)
    ekw = keyword_ct["c1"].tobytes()
    if attack_type == "bad_sig":
        q = Query(encrypted_keyword=ekw, signature=b"x" * 32, epoch=epoch)
    elif attack_type == "wrong_epoch":
        wrong_ep = epoch + 99
        q = Query(encrypted_keyword=ekw, signature=b"", epoch=wrong_ep)
        msg = validator.serialize(user_id, q)
        q.signature = sig.sign(msg, b"fake_key_xxxxxxxxxxxxxxxxxxxxx")
    elif attack_type == "flood":
        q = Query(encrypted_keyword=ekw, signature=b"flood" * 6 + b"ff", epoch=epoch)
    else:
        q = Query(encrypted_keyword=ekw, signature=b"y" * 32, epoch=epoch)
    return q


def train_ml_engine(verbose=True):
    """Train the ML trust engine. Returns trained engine."""
    data_path = os.path.join(os.path.dirname(__file__) or ".", "training_data.csv")
    if not os.path.exists(data_path):
        if verbose:
            print("  Generating training data...", flush=True)
        samples = generate_full_dataset()
        save_dataset(samples, data_path)

    from attack_simulator import load_dataset
    X, y, _ = load_dataset(data_path)
    engine = MLTrustEngine(n_estimators_iso=80, n_estimators_rf=80)
    metrics = engine.train(X, y, verbose=verbose)
    return engine, metrics


def run_ml_trust_comparison(n=64, num_queries=20, num_attacks=15, tree_depth=3):
    """
    Compare static trust scoring vs ML-enhanced trust scoring.
    Returns dict with comparison metrics.
    """
    params, system, ops, sig, sk_user, pk_user = _setup_system(n, tree_depth)
    user_id = "TestDevice"
    epoch = 1
    nodes = list(range(min(epoch + 2, 2 ** tree_depth)))

    # Train ML engine
    ml_engine, ml_train_metrics = train_ml_engine(verbose=False)

    # ========== STATIC TRUST TEST ==========
    tm_static = TrustManager()
    validator_static = QueryValidator(tm_static, sig, params)

    static_tp, static_fp, static_tn, static_fn = 0, 0, 0, 0

    # Legitimate queries (should be accepted)
    for _ in range(num_queries):
        q, _ = _make_query(ops, validator_static, sig, sk_user, user_id, epoch)
        accepted = validator_static.validate_static(user_id, q, pk_user)
        if accepted:
            static_tn += 1  # correctly accepted normal
        else:
            static_fp += 1  # wrongly rejected normal

    # Attack queries (should be rejected)
    attack_types = ["bad_sig", "wrong_epoch", "flood", "bad_sig"]
    for i in range(num_attacks):
        atype = attack_types[i % len(attack_types)]
        q = _make_malicious_query(ops, validator_static, sig, user_id, epoch, atype)
        # Reset trust so attacker isn't blocked by accumulated penalties
        tm_static.db[user_id] = 5.0
        accepted = validator_static.validate_static(user_id, q, pk_user)
        if not accepted:
            static_tp += 1  # correctly rejected attack
        else:
            static_fn += 1  # wrongly accepted attack

    # ========== ML TRUST TEST ==========
    tm_ml = TrustManager(ml_engine=ml_engine)
    fe = FeatureExtractor(system_epoch=epoch)
    validator_ml = QueryValidator(tm_ml, sig, params, feature_extractor=fe)

    ml_tp, ml_fp, ml_tn, ml_fn = 0, 0, 0, 0
    ml_inference_times = []

    # Legitimate queries
    for _ in range(num_queries):
        fe.reset_user(user_id)
        tm_ml.db[user_id] = 5.0
        q, _ = _make_query(ops, validator_ml, sig, sk_user, user_id, epoch)
        t0 = time.perf_counter()
        accepted = validator_ml.validate(user_id, q, pk_user)
        ml_inference_times.append(time.perf_counter() - t0)
        if accepted:
            ml_tn += 1
        else:
            ml_fp += 1

    # Attack queries with realistic features
    for i in range(num_attacks):
        fe.reset_user("attacker")
        tm_ml.db["attacker"] = 5.0
        atype = attack_types[i % len(attack_types)]

        # Simulate attack behavior in feature extractor
        for _ in range(5):
            fake_q = Query(b"x" * 100, b"", epoch + 50)
            fe.extract("attacker", fake_q, sig_valid=False)

        q = _make_malicious_query(ops, validator_ml, sig, "attacker", epoch, atype)
        t0 = time.perf_counter()
        accepted = validator_ml.validate("attacker", q, pk_user)
        ml_inference_times.append(time.perf_counter() - t0)
        if not accepted:
            ml_tp += 1
        else:
            ml_fn += 1

    # Compute metrics
    def safe_div(a, b):
        return a / b if b > 0 else 0.0

    def compute_metrics(tp, fp, tn, fn):
        total = tp + fp + tn + fn
        accuracy = safe_div(tp + tn, total)
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)
        fpr = safe_div(fp, fp + tn)
        return {
            "accuracy": accuracy, "precision": precision,
            "recall": recall, "f1": f1,
            "fpr": fpr, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        }

    static_metrics = compute_metrics(static_tp, static_fp, static_tn, static_fn)
    ml_metrics = compute_metrics(ml_tp, ml_fp, ml_tn, ml_fn)

    avg_ml_time = sum(ml_inference_times) / len(ml_inference_times) if ml_inference_times else 0

    return {
        "static": static_metrics,
        "ml": ml_metrics,
        "ml_train_metrics": ml_train_metrics,
        "ml_avg_inference_ms": avg_ml_time * 1000,
        "num_queries": num_queries,
        "num_attacks": num_attacks,
    }


def print_ml_comparison(results):
    """Print comparison table."""
    s = results["static"]
    m = results["ml"]
    print("\n" + "=" * 70, flush=True)
    print("  ML Trust Scoring -- Static vs ML-Enhanced Comparison", flush=True)
    print("=" * 70, flush=True)
    print(f"  {'Metric':<25} {'Static':>15} {'ML-Enhanced':>15}", flush=True)
    print("  " + "-" * 60, flush=True)
    for key in ["accuracy", "precision", "recall", "f1", "fpr"]:
        print(f"  {key:<25} {s[key]:>15.4f} {m[key]:>15.4f}", flush=True)
    print("  " + "-" * 60, flush=True)
    print(f"  {'TP (attacks caught)':<25} {s['tp']:>15} {m['tp']:>15}", flush=True)
    print(f"  {'FP (false alarms)':<25} {s['fp']:>15} {m['fp']:>15}", flush=True)
    print(f"  {'TN (legit accepted)':<25} {s['tn']:>15} {m['tn']:>15}", flush=True)
    print(f"  {'FN (attacks missed)':<25} {s['fn']:>15} {m['fn']:>15}", flush=True)
    print("  " + "-" * 60, flush=True)
    print(f"  ML Avg Inference Time: {results['ml_avg_inference_ms']:.3f} ms", flush=True)
    print("=" * 70, flush=True)

    # ML Training metrics
    tm = results["ml_train_metrics"]
    print("\n  ML Model Training Performance:", flush=True)
    print(f"    Isolation Forest  -- Acc: {tm['iso_accuracy']:.4f}  F1: {tm['iso_f1']:.4f}", flush=True)
    print(f"    Random Forest     -- Acc: {tm['rf_accuracy']:.4f}  F1: {tm['rf_f1']:.4f}", flush=True)
    print(f"    Combined          -- Acc: {tm['combined_accuracy']:.4f}  F1: {tm['combined_f1']:.4f}", flush=True)
    print("=" * 70, flush=True)


def save_ml_comparison_csv(results, path="ML_Trust_Comparison.csv"):
    """Save comparison results to CSV."""
    rows = []
    for label, data in [("static", results["static"]), ("ml_enhanced", results["ml"])]:
        row = {"method": label}
        row.update(data)
        if label == "ml_enhanced":
            row["avg_inference_ms"] = results["ml_avg_inference_ms"]
        rows.append(row)
    fieldnames = list(rows[0].keys())
    if "avg_inference_ms" not in fieldnames:
        fieldnames.append("avg_inference_ms")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"  Saved: {path}", flush=True)


def save_ml_results_json(results, path="ml_trust_results.json"):
    """Save full ML results as JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved: {path}", flush=True)


if __name__ == "__main__":
    print("=" * 70, flush=True)
    print("  ML Trust Scoring Simulation", flush=True)
    print("=" * 70, flush=True)

    print("\n  Running ML vs Static trust comparison...\n", flush=True)
    results = run_ml_trust_comparison(n=64, num_queries=20, num_attacks=15)
    print_ml_comparison(results)
    save_ml_comparison_csv(results)
    save_ml_results_json(results)
    print("\n  Done.", flush=True)
