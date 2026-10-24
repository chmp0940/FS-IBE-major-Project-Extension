"""
ML Trust Simulation — Benchmarks ML-enhanced trust scoring vs static scoring.
Integrates with the existing simulation pipeline.

Supports the 6-model ML Trust Engine:
  Isolation Forest, Random Forest, SVM, KNN, Decision Tree, MLP
"""
import sys, time, os, csv, json
import numpy as np

import lattice_infrastructure
sys.modules["LatticeCrypto"] = lattice_infrastructure

import lattice_infrastructure as P1
from forward_security import UserOps
from Trust_Model import (TrustManager, DilithiumStub, Query, QueryValidator,
                         match_query_to_data, FeatureExtractor)
from ml_trust_engine import MLTrustEngine, SUPERVISED_MODEL_NAMES
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


def _make_malicious_query(
    ops, validator, sig, user_id, epoch, attack_type="bad_sig", signing_key=None
):
    """Generate a malicious query, including behavior-only attacks with valid signatures."""
    keyword_ct = ops.Encrypt(user_id, epoch, 1)
    ekw = keyword_ct["c1"].tobytes()
    if attack_type == "bad_sig":
        q = Query(encrypted_keyword=ekw, signature=b"x" * 32, epoch=epoch)
    elif attack_type == "valid_wrong_epoch":
        wrong_ep = epoch + 99
        q = Query(encrypted_keyword=ekw, signature=b"", epoch=wrong_ep)
        msg = validator.serialize(user_id, q)
        q.signature = sig.sign(msg, signing_key)
    elif attack_type == "valid_replay":
        # A replayed query carries an old epoch, while retaining a valid
        # signature. This makes the replay behavior observable to the feature
        # extractor instead of treating it as ordinary current-epoch traffic.
        old_epoch = max(0, epoch - 1)
        keyword_ct = ops.Encrypt(user_id, old_epoch, 1)
        q = Query(
            encrypted_keyword=keyword_ct["c1"].tobytes(),
            signature=b"",
            epoch=old_epoch,
        )
        msg = validator.serialize(user_id, q)
        q.signature = sig.sign(msg, signing_key)
    elif attack_type in ("valid_flood", "valid_stealth"):
        q = Query(encrypted_keyword=ekw, signature=b"", epoch=epoch)
        msg = validator.serialize(user_id, q)
        q.signature = sig.sign(msg, signing_key)
    else:
        q = Query(encrypted_keyword=ekw, signature=b"y" * 32, epoch=epoch)
    return q


def train_ml_engine(verbose=True):
    """Train the ML trust engine (all 6 models). Returns trained engine."""
    data_path = os.path.join(os.path.dirname(__file__) or ".", "training_data.csv")
    if not os.path.exists(data_path):
        if verbose:
            print("  Generating training data...", flush=True)
        samples = generate_full_dataset()
        save_dataset(samples, data_path)

    from attack_simulator import load_dataset
    X, y, _ = load_dataset(data_path)
    # The benchmark creates many short-lived model evaluations. A single
    # worker avoids loky/resource-tracker warnings on Windows and is faster
    # than repeatedly starting process pools for these small predictions.
    engine = MLTrustEngine(n_estimators_iso=80, n_estimators_rf=80, n_jobs=1)
    metrics = engine.train(X, y, verbose=verbose)
    return engine, metrics


def run_ml_trust_comparison(n=64, num_queries=20, num_attacks=15, tree_depth=3):
    """
    Compare static trust scoring vs ML-enhanced trust scoring.

    Each sample is evaluated in an independent episode. Static and ML
    validators receive the same traffic shape, and behavioral history is
    created only by actual queries processed through the validator.
    Returns dict with comparison metrics.
    """
    params, system, ops, sig, sk_user, pk_user = _setup_system(n, tree_depth)
    epoch = 1

    # Train ML engine
    ml_engine, ml_train_metrics = train_ml_engine(verbose=False)

    attack_types = [
        "valid_flood", "valid_wrong_epoch", "valid_replay",
        "valid_stealth", "bad_sig",
    ]

    def safe_div(a, b):
        return a / b if b > 0 else 0.0

    def compute_metrics(tp, fp, tn, fn):
        total = tp + fp + tn + fn
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        return {
            "accuracy": safe_div(tp + tn, total),
            "precision": precision,
            "recall": recall,
            "f1": safe_div(2 * precision * recall, precision + recall),
            "fpr": safe_div(fp, fp + tn),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        }

    def run_episode(atype, use_ml):
        tm = TrustManager(ml_engine=ml_engine if use_ml else None)
        fe = FeatureExtractor(system_epoch=epoch) if use_ml else None
        validator = QueryValidator(tm, sig, params, feature_extractor=fe)
        uid = "attacker" if atype != "normal" else "normal-device"
        validator_fn = validator.validate if use_ml else validator.validate_static
        replay_query = None
        if atype == "valid_replay":
            replay_query = _make_malicious_query(
                ops, validator, sig, uid, epoch, atype, signing_key=sk_user
            )

        # Establish history through the same public validation path for both
        # models. This is traffic, not direct feature injection.
        if atype == "valid_flood":
            # Create an actual burst, not merely five ordinary queries.
            history_count = 40
        elif atype in ("bad_sig", "valid_wrong_epoch", "valid_replay"):
            history_count = 8
        else:
            history_count = 2
        for _ in range(history_count):
            if atype == "bad_sig":
                history_query = _make_malicious_query(
                    ops, validator, sig, uid, epoch, "bad_sig", signing_key=sk_user
                )
            elif atype == "valid_wrong_epoch":
                history_query = _make_malicious_query(
                    ops, validator, sig, uid, epoch, atype, signing_key=sk_user
                )
            elif atype == "valid_replay":
                history_query = replay_query
            else:
                history_query, _ = _make_query(ops, validator, sig, sk_user, uid, epoch)
            validator_fn(uid, history_query, pk_user)

        # Keep the behavioral history but isolate the target decision from
        # trust-score exhaustion caused by the setup traffic.
        tm.db[uid] = 5.0
        if atype == "normal":
            target, _ = _make_query(ops, validator, sig, sk_user, uid, epoch)
            expected_attack = False
        elif atype == "valid_replay":
            target = replay_query
            expected_attack = True
        else:
            target = _make_malicious_query(
                ops, validator, sig, uid, epoch, atype, signing_key=sk_user
            )
            expected_attack = True

        started = time.perf_counter()
        accepted = validator_fn(uid, target, pk_user)
        elapsed = (time.perf_counter() - started) * 1000
        predicted_attack = not accepted
        return expected_attack, predicted_attack, elapsed

    def evaluate_model(use_ml):
        tp = fp = tn = fn = 0
        timings = []
        by_attack = {}

        for _ in range(num_queries):
            expected, predicted, elapsed = run_episode("normal", use_ml)
            timings.append(elapsed)
            if predicted:
                fp += 1
            else:
                tn += 1

        for i in range(num_attacks):
            atype = attack_types[i % len(attack_types)]
            expected, predicted, elapsed = run_episode(atype, use_ml)
            timings.append(elapsed)
            counters = by_attack.setdefault(atype, [0, 0])
            counters[0] += 1
            counters[1] += int(predicted)
            if predicted:
                tp += 1
            else:
                fn += 1

        attack_metrics = {
            name: {
                "samples": values[0],
                "detected": values[1],
                "recall": safe_div(values[1], values[0]),
            }
            for name, values in by_attack.items()
        }
        return compute_metrics(tp, fp, tn, fn), attack_metrics, timings

    static_metrics, static_by_attack, _ = evaluate_model(use_ml=False)
    ml_metrics, ml_by_attack, _ = evaluate_model(use_ml=True)

    return {
        "static": static_metrics,
        "ml": ml_metrics,
        "ml_train_metrics": ml_train_metrics,
        "ml_avg_inference_ms": None,
        "static_by_attack": static_by_attack,
        "ml_by_attack": ml_by_attack,
        "num_queries": num_queries,
        "num_attacks": num_attacks,
        "evaluation": {
            "independent_episodes": True,
            "feature_injection": False,
            "query_recorded_once": True,
            "inference_timing": "not reported; benchmark process timing is environment-dependent",
        },
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
    if results["ml_avg_inference_ms"] is not None:
        print(f"  ML Avg Inference Time: {results['ml_avg_inference_ms']:.3f} ms", flush=True)
    else:
        print("  ML Avg Inference Time: not measured", flush=True)
    print("=" * 70, flush=True)

    # ML Training metrics — all 6 models + combined
    tm = results["ml_train_metrics"]
    model_display = {
        "iso": "Isolation Forest",
        "random_forest": "Random Forest",
        "svm": "SVM",
        "knn": "KNN",
        "decision_tree": "Decision Tree",
        "mlp": "MLP (Neural Net)",
        "combined": "Combined Ensemble",
    }
    print("\n  ML Model Training Performance (All 6 Models):", flush=True)
    print(f"  {'Model':<22} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}", flush=True)
    print("  " + "-" * 62, flush=True)
    for key, display_name in model_display.items():
        acc = tm.get(f"{key}_accuracy", 0)
        prec = tm.get(f"{key}_precision", 0)
        rec = tm.get(f"{key}_recall", 0)
        f1 = tm.get(f"{key}_f1", 0)
        print(f"  {display_name:<22} {acc:>10.4f} {prec:>10.4f} {rec:>10.4f} {f1:>10.4f}", flush=True)
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
    print("  ML Trust Scoring Simulation (6 Models)", flush=True)
    print("=" * 70, flush=True)

    print("\n  Running ML vs Static trust comparison...\n", flush=True)
    results = run_ml_trust_comparison(n=64, num_queries=20, num_attacks=15)
    print_ml_comparison(results)
    save_ml_comparison_csv(results)
    save_ml_results_json(results)
    print("\n  Done.", flush=True)
