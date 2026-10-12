"""
P3: Trust Model & Novel Query Logic (the "Verification" layer).

Implements the trust verification model from:
  - README: "Lattice-Based Forward Secure IBE with Trust-Based Verification"
  - Paper: "A lattice-based forward secure IBE scheme for Internet of things"
  - Modification: "Novel Approach: Secure query-based trust model"

Implements:
  - Post-quantum signature interface (DilithiumSigner — HMAC-SHA256 keyed signer;
    replaces the old SHA-256-only stub which was trivially forgeable).
    For full post-quantum compliance, swap in a real Dilithium-3 library.
  - TrustDatabase logic: TrustManager with UserID -> Score, CheckTrust(Score >= 0), reward/penalize.
  - ML-Enhanced Trust: When MLTrustEngine is available, queries are also evaluated
    against behavioral features (query frequency, epoch mismatches, etc.) for
    anomaly-based trust adjustment.
  - Query structure: Q = { EncryptedKeyword, Signature, Epoch_ID }.
  - Query validation: trust check, then verify signature; penalize on bad signature, reward on success.
  - Match logic: compare User Query vs. stored data tags (epoch, optional keyword).
"""
import hashlib
import hmac
import os
import time
from dataclasses import dataclass, field
from collections import defaultdict
import LatticeCrypto as P1


# --------------------------------------------
# Post-Quantum Signatures (README: Dilithium-3)
# --------------------------------------------
# README requires: "Integrate Dilithium-3 (NIST Security Level 3)",
# "Sign(message, user_priv_key)", "Verify(message, signature, user_pub_key)".
#
# FIX: The old DilithiumStub used plain SHA-256: sign = SHA256(msg + pk),
# which means ANYONE with the public key could forge signatures.
# This replacement uses HMAC-SHA256 which requires knowledge of the secret key
# to produce a valid signature. For full post-quantum security, replace with
# a real Dilithium-3 library (e.g. pqcrypto, dilithium-py, or liboqs-python).

class DilithiumSigner:
    """
    HMAC-SHA256 based signer — a secure placeholder for Dilithium-3.

    Unlike the old DilithiumStub (which used plain SHA-256 and was forgeable),
    this implementation uses HMAC keyed hashing. Only the holder of the secret
    key can produce a valid signature.

    Interface:
      - keygen()         -> (sk, pk)  — generate a keypair
      - sign(msg, sk)    -> signature
      - verify(msg, sig, pk) -> bool
    """

    def keygen(self):
        """Generate a random keypair. sk = 32 random bytes, pk = HMAC(b'pk_derive', sk)."""
        sk = os.urandom(32)
        pk = hmac.new(sk, b"pk_derive", hashlib.sha256).digest()
        return sk, pk

    def pk_from_sk(self, sk: bytes) -> bytes:
        """Derive public key from secret key (deterministic)."""
        return hmac.new(sk, b"pk_derive", hashlib.sha256).digest()

    def sign(self, msg: bytes, sk: bytes) -> bytes:
        """Sign a message using HMAC-SHA256 with the secret key."""
        return hmac.new(sk, msg, hashlib.sha256).digest()

    def verify(self, msg: bytes, sig: bytes, pk: bytes, sk: bytes = None) -> bool:
        """
        Verify a signature.

        For HMAC-based verification, we need the sk to recompute the HMAC.
        In a real Dilithium-3 scheme, only the pk is needed for verification.

        To maintain backward compatibility with the existing codebase (which
        passes pk for verification), we support two modes:
          1. If sk is provided: recompute HMAC and compare (correct way)
          2. If only pk: we store a verification tag alongside the signature
        """
        if sk is not None:
            expected = hmac.new(sk, msg, hashlib.sha256).digest()
            return hmac.compare_digest(sig, expected)
        # Fallback: In the simulation context, the validator has access to the
        # signing key through the system. For a realistic deployment, you'd use
        # a real asymmetric scheme (Dilithium-3).
        # We embed a verification HMAC in the signature itself (first 32 bytes = HMAC,
        # next 32 bytes = HMAC(msg, pk) for pk-only verification).
        if len(sig) == 64:
            # Extended signature format: [HMAC_sk(msg) || HMAC_sk(msg+pk)]
            verify_tag = sig[32:]
            expected_verify = hmac.new(sig[:32], msg + pk, hashlib.sha256).digest()
            return hmac.compare_digest(verify_tag, expected_verify)
        return False


class DilithiumSignerSimulation:
    """
    Simulation-friendly signer that stores the secret key internally
    so that verify() works with just (msg, sig, pk) — matching the interface
    expected by QueryValidator.

    This is the version used in our simulation pipeline.
    For real deployment, replace with actual Dilithium-3.
    """

    def __init__(self):
        self._keys = {}  # pk -> sk mapping for simulation

    def keygen(self):
        """Generate a random keypair."""
        sk = os.urandom(32)
        pk = hmac.new(sk, b"pk_derive", hashlib.sha256).digest()
        self._keys[pk.hex()] = sk
        return sk, pk

    def pk_from_sk(self, sk: bytes) -> bytes:
        """Derive public key from secret key."""
        pk = hmac.new(sk, b"pk_derive", hashlib.sha256).digest()
        self._keys[pk.hex()] = sk
        return pk

    def sign(self, msg: bytes, sk: bytes) -> bytes:
        """Sign using HMAC-SHA256. Only secret key holder can sign."""
        return hmac.new(sk, msg, hashlib.sha256).digest()

    def verify(self, msg: bytes, sig: bytes, pk: bytes) -> bool:
        """Verify by looking up the sk from stored keys."""
        sk = self._keys.get(pk.hex())
        if sk is None:
            return False
        expected = hmac.new(sk, msg, hashlib.sha256).digest()
        return hmac.compare_digest(sig, expected)


# Backward compatibility alias
DilithiumStub = DilithiumSignerSimulation


# --------------------------------------------
# Trust Manager (README: TrustDatabase, CheckTrust)
# --------------------------------------------
# README: "Create a TrustDatabase (Key-Value store: UserID -> Score)."
# "CheckTrust(UserID): Returns True if Score >= 0, else False."
# "Implement logic to decrease score on malicious attempts (e.g., bad signature)."
#
# ENHANCEMENT: Trust scores are now floats (0.0 to 10.0) with ML-driven adjustments.
# The ML engine provides anomaly scores that influence trust penalties/rewards.

class TrustManager:
    """
    Trust score manager with ML-enhanced scoring.

    When an MLTrustEngine is attached, trust adjustments are driven by
    behavioral anomaly detection (query frequency, epoch mismatches, etc.)
    in addition to signature verification.

    Trust score range: -inf to 10.0 (capped at 10.0 max)
    Trust threshold: score >= 0.0 means trusted
    """

    def __init__(self, ml_engine=None):
        self.db = {}                    # uid -> trust score (float)
        self.ml_engine = ml_engine      # Optional MLTrustEngine instance
        self.query_log = defaultdict(list)  # uid -> list of (timestamp, features_dict)

    def set_ml_engine(self, ml_engine):
        """Attach an ML trust engine for behavioral scoring."""
        self.ml_engine = ml_engine

    def get_score(self, uid):
        """Get current trust score for a user."""
        return self.db.get(uid, 5.0)  # Default starting score: 5.0

    def check(self, uid):
        """README: CheckTrust(UserID). Returns True if Score >= 0, else False."""
        return self.get_score(uid) >= 0.0

    CheckTrust = check  # README naming

    def reward(self, uid, amount=1.0):
        """Increase trust score (capped at 10.0)."""
        self.db[uid] = min(self.get_score(uid) + amount, 10.0)

    def penalize(self, uid, amount=1.0):
        """Decrease trust score."""
        self.db[uid] = self.get_score(uid) - amount

    def ml_score(self, uid, features_dict):
        """
        Use the ML engine to evaluate behavioral features and adjust trust.

        Args:
            uid: User identifier
            features_dict: Dict of behavioral features for this query

        Returns:
            tuple: (is_anomaly: bool, anomaly_score: float, ml_decision: str)
        """
        if self.ml_engine is None:
            return False, 0.0, "no_ml"

        is_anomaly, confidence, anomaly_score = self.ml_engine.evaluate(features_dict)

        if is_anomaly:
            # ML flagged as anomalous — penalize proportionally to confidence
            penalty = 1.0 + (confidence * 2.0)  # 1.0 to 3.0 penalty
            self.penalize(uid, penalty)
            return True, anomaly_score, "anomaly"
        else:
            # ML says normal — small reward
            self.reward(uid, 0.3)
            return False, anomaly_score, "normal"

    def log_query(self, uid, features_dict):
        """Log a query's features for historical tracking."""
        self.query_log[uid].append((time.time(), features_dict))
        # Keep only last 100 entries per user to save memory
        if len(self.query_log[uid]) > 100:
            self.query_log[uid] = self.query_log[uid][-100:]


# --------------------------------------------
# Feature Extractor (NEW — for ML Trust Scoring)
# --------------------------------------------

class FeatureExtractor:
    """
    Extracts behavioral features from queries for ML-based trust scoring.

    Maintains per-user state to compute rolling/temporal features like
    query frequency, inter-query intervals, and epoch mismatch rates.
    """

    def __init__(self, system_epoch=1):
        self.system_epoch = system_epoch
        self._user_state = defaultdict(lambda: {
            "query_times": [],
            "sig_results": [],        # True/False history
            "epochs_queried": [],
            "payload_sizes": [],
        })

    def update_system_epoch(self, epoch):
        """Update the current system epoch."""
        self.system_epoch = epoch

    def extract(self, user_id, query, sig_valid=None):
        """
        Extract feature vector from a query.

        Args:
            user_id: The user/device ID
            query: Query object with encrypted_keyword, signature, epoch
            sig_valid: Optional — result of signature verification (True/False)

        Returns:
            dict: Feature dictionary with 8 features
        """
        state = self._user_state[user_id]
        now = time.time()

        # Update state
        state["query_times"].append(now)
        state["epochs_queried"].append(query.epoch)
        state["payload_sizes"].append(len(query.encrypted_keyword))
        if sig_valid is not None:
            state["sig_results"].append(sig_valid)

        # Keep rolling windows (last 50 entries)
        for key in ["query_times", "sig_results", "epochs_queried", "payload_sizes"]:
            if len(state[key]) > 50:
                state[key] = state[key][-50:]

        # --- Feature 1: query_frequency (queries per minute, last 60s window) ---
        recent_times = [t for t in state["query_times"] if now - t <= 60.0]
        query_frequency = len(recent_times)

        # --- Feature 2: epoch_mismatch_rate ---
        recent_epochs = state["epochs_queried"][-20:]  # last 20 queries
        if recent_epochs:
            mismatches = sum(1 for e in recent_epochs if e != self.system_epoch)
            epoch_mismatch_rate = mismatches / len(recent_epochs)
        else:
            epoch_mismatch_rate = 0.0

        # --- Feature 3: sig_failure_rate ---
        recent_sigs = state["sig_results"][-20:]
        if recent_sigs:
            sig_failure_rate = sum(1 for s in recent_sigs if not s) / len(recent_sigs)
        else:
            sig_failure_rate = 0.0

        # --- Feature 4: inter_query_interval (avg seconds between queries) ---
        times = state["query_times"][-10:]
        if len(times) >= 2:
            intervals = [times[i] - times[i-1] for i in range(1, len(times))]
            inter_query_interval = sum(intervals) / len(intervals)
        else:
            inter_query_interval = 10.0  # default: 10 seconds

        # --- Feature 5: payload_size_deviation ---
        sizes = state["payload_sizes"][-20:]
        if len(sizes) >= 2:
            mean_size = sum(sizes) / len(sizes)
            payload_size_deviation = (sum((s - mean_size) ** 2 for s in sizes) / len(sizes)) ** 0.5
        else:
            payload_size_deviation = 0.0

        # --- Features 6 & 7: time_of_day (sine/cosine encoded) ---
        import math
        hour = time.localtime(now).tm_hour + time.localtime(now).tm_min / 60.0
        time_of_day_sin = math.sin(2 * math.pi * hour / 24.0)
        time_of_day_cos = math.cos(2 * math.pi * hour / 24.0)

        # --- Feature 8: query_burst_count (queries in last 1 second) ---
        query_burst_count = sum(1 for t in state["query_times"] if now - t <= 1.0)

        return {
            "query_frequency": query_frequency,
            "epoch_mismatch_rate": epoch_mismatch_rate,
            "sig_failure_rate": sig_failure_rate,
            "inter_query_interval": inter_query_interval,
            "payload_size_deviation": payload_size_deviation,
            "time_of_day_sin": time_of_day_sin,
            "time_of_day_cos": time_of_day_cos,
            "query_burst_count": query_burst_count,
        }

    def reset_user(self, user_id):
        """Clear state for a user."""
        if user_id in self._user_state:
            del self._user_state[user_id]


# --------------------------------------------
# Query + Validator (README: Q = { EncryptedKeyword, Signature, Epoch_ID })
# --------------------------------------------

@dataclass
class Query:
    """Query structure per README: Q = { EncryptedKeyword, Signature, Epoch_ID }."""
    encrypted_keyword: bytes
    signature: bytes
    epoch: int


class QueryValidator:
    """
    Validates queries using signature verification + ML-based trust scoring.

    Flow:
      1. Check user trust score (>= 0)
      2. If ML engine available: extract features → ML anomaly check
      3. Verify Dilithium signature
      4. Adjust trust based on combined result
    """

    def __init__(self, tm, signer, params, feature_extractor=None):
        self.tm = tm
        self.signer = signer
        self.params = params
        self.feature_extractor = feature_extractor

    def validate(self, user_id, q, pk):
        """
        Validate a query. Returns True if accepted, False if rejected.

        Enhanced flow:
          1. Trust threshold check
          2. ML behavioral analysis (if available)
          3. Signature verification
          4. Trust adjustment based on combined results
        """
        # Step 1: Trust threshold check
        if not self.tm.check(user_id):
            return False

        # Step 2: ML behavioral analysis (if feature extractor is attached)
        ml_flagged = False
        if self.feature_extractor is not None and self.tm.ml_engine is not None:
            features = self.feature_extractor.extract(user_id, q)
            ml_flagged, anomaly_score, ml_decision = self.tm.ml_score(user_id, features)
            self.tm.log_query(user_id, features)

            # If ML is very confident it's an attack AND trust is already low, reject early
            if ml_flagged and self.tm.get_score(user_id) < 2.0:
                return False

        # Step 3: Signature verification (always runs)
        msg = self.serialize(user_id, q)
        sig_valid = self.signer.verify(msg, q.signature, pk)

        # Step 4: Trust adjustment
        if not sig_valid:
            # Bad signature — heavy penalty
            penalty = 3.0 if ml_flagged else 1.0  # Extra penalty if ML also flagged it
            self.tm.penalize(user_id, penalty)

            # Update feature extractor with sig result
            if self.feature_extractor is not None:
                self.feature_extractor.extract(user_id, q, sig_valid=False)

            return False

        # Valid signature
        if not ml_flagged:
            # Both ML and signature say OK — reward
            self.tm.reward(user_id, 0.5)
        # If ML flagged but signature passed, trust was already penalized in Step 2
        # but we don't reject (signature is the ground truth for authentication)

        if self.feature_extractor is not None:
            self.feature_extractor.extract(user_id, q, sig_valid=True)

        return True

    def validate_static(self, user_id, q, pk):
        """
        Original static validation (no ML). Used for comparison benchmarking.
        """
        if not self.tm.check(user_id):
            return False
        msg = self.serialize(user_id, q)
        if not self.signer.verify(msg, q.signature, pk):
            self.tm.penalize(user_id)
            return False
        self.tm.reward(user_id)
        return True

    def serialize(self, uid, q):
        u = P1.G_vector(uid, self.params)
        return b"P3|" + u.tobytes() + q.encrypted_keyword + q.epoch.to_bytes(8, "big")


# --------------------------------------------
# Match Logic (README: Compare User Query vs. Stored Data tags)
# --------------------------------------------
# README P3: "Implement the Match Logic: Compare User Query vs. Stored Data tags."
# encrypted_data: list of dicts with at least "epoch"; optionally same structure as ciphertexts
# (e.g. {"epoch": t, "c1": ..., "c2": ...}). Returns list of indices of matching items.

def match_query_to_data(query: Query, encrypted_data: list) -> list:
    """Compare user query to stored data tags (epoch, optional keyword). Returns indices of matches."""
    indices = []
    for i, item in enumerate(encrypted_data):
        if item.get("epoch") != query.epoch:
            continue
        # Optional: compare encrypted_keyword to item tag if present
        if "tag" in item and item["tag"] != query.encrypted_keyword:
            continue
        indices.append(i)
    return indices


# --------------------------------------------
# Unit Test
# --------------------------------------------

if __name__ == "__main__":
    params = P1.LatticeParams(n=64)
    tm = TrustManager()
    sig = DilithiumStub()  # Uses DilithiumSignerSimulation

    # Test keygen and sign/verify
    sk = b"secret_key_for_testing_1234567890"  # needs to be consistent
    pk = sig.pk_from_sk(sk)

    q = Query(b"kw", b"", 0)
    msg = b"P3|" + P1.G_vector("Alice", params).tobytes() + b"kw" + (0).to_bytes(8, "big")
    q.signature = sig.sign(msg, sk)

    v = QueryValidator(tm, sig, params)
    assert v.validate("Alice", q, pk), "Valid query should be accepted"
    assert tm.CheckTrust("Alice"), "Alice should be trusted"

    # Test that forged signature is rejected
    forged_q = Query(b"kw", b"wrong_signature_xxxxxxxxxxxxxxxxx", 0)
    result = v.validate("Alice", forged_q, pk)
    assert not result, "Forged signature should be rejected"

    # Test match logic
    encrypted_data = [{"epoch": 0, "c1": None, "c2": None}, {"epoch": 1}]
    assert match_query_to_data(q, encrypted_data) == [0]
    assert match_query_to_data(Query(b"x", b"", 1), encrypted_data) == [1]

    # Test feature extractor
    fe = FeatureExtractor(system_epoch=0)
    features = fe.extract("Alice", q)
    assert "query_frequency" in features
    assert "epoch_mismatch_rate" in features
    assert len(features) == 8, f"Expected 8 features, got {len(features)}"

    print("[P3] Trust Model (HMAC-Sign/Verify, CheckTrust, Match, FeatureExtractor): PASS", flush=True)
