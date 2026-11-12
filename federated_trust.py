"""
Federated Trust Client (IoT Gateway Node)
Trains a local behavioral anomaly detection model on gateway-specific IoT traffic,
quantizes model parameters, encrypts updates via Lattice fs-IBE (Dual Regev),
and signs the payload using CRYSTALS-Dilithium-3.
"""

import time
from typing import Optional
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import lattice_infrastructure as P1
from forward_security import UserOps
from Trust_Model import DilithiumSigner


class FederatedGatewayClient:
    """
    Represents an IoT Edge Gateway participating in Federated Learning.
    Holds local IoT query logs, trains local anomaly detection weights,
    and transmits fs-IBE encrypted + Dilithium-3 signed updates.
    """

    def __init__(
        self,
        gateway_id: str,
        system: dict,
        signer: DilithiumSigner,
        n_features: int = 8,
        shared_scaler: Optional[StandardScaler] = None,
    ):
        self.gateway_id = gateway_id
        self.system = system
        self.ops = UserOps(system)
        self.signer = signer
        self.n_features = n_features
        self.scaler = shared_scaler if shared_scaler is not None else StandardScaler()
        self._uses_shared_scaler = shared_scaler is not None

        # Generate real Dilithium-3 keypair for this gateway
        self.sk_sig, self.pk_sig = self.signer.keygen()

        # Local parametric model for FedAvg (steady convergence across rounds)
        self.local_model = SGDClassifier(
            loss="log_loss", max_iter=1, learning_rate="constant", eta0=0.0015, random_state=42
        )
        # Local unsupervised Isolation Forest for zero-day anomalies
        self.iso_forest = IsolationForest(n_estimators=50, contamination=0.15, random_state=42)

        # Initialize weight shapes: coef_ (1, n_features) and intercept_ (1,)
        self.local_model.coef_ = np.zeros((1, n_features), dtype=np.float64)
        self.local_model.intercept_ = np.zeros((1,), dtype=np.float64)
        self.local_model.classes_ = np.array([0, 1])

        self.is_initialized = False
        self.num_samples = 0

    def set_global_weights(self, coef: np.ndarray, intercept: np.ndarray):
        """Receive and load aggregated global model weights from the FL server."""
        self.local_model.coef_ = coef.copy()
        self.local_model.intercept_ = intercept.copy()
        self.is_initialized = True

    def train_local(self, X_local: np.ndarray, y_local: np.ndarray, epochs: int = 5):
        """Train local model on gateway's private IoT traffic."""
        self.num_samples = len(X_local)
        if self._uses_shared_scaler:
            X_scaled = self.scaler.transform(X_local)
        else:
            X_scaled = self.scaler.fit_transform(X_local)

        # Fit unsupervised Isolation Forest on normal traffic
        normal_mask = (y_local == 0)
        if np.any(normal_mask):
            self.iso_forest.fit(X_scaled[normal_mask])

        # Perform local SGD epochs starting from current global weights
        for _ in range(epochs):
            self.local_model.partial_fit(X_scaled, y_local, classes=np.array([0, 1]))

        self.is_initialized = True
        preds = self.local_model.predict(X_scaled)
        return {
            "gateway_id": self.gateway_id,
            "samples": self.num_samples,
            "local_acc": accuracy_score(y_local, preds),
            "local_f1": f1_score(y_local, preds, zero_division=0),
        }

    @staticmethod
    def _quantize_weights_to_bits(coef: np.ndarray, intercept: np.ndarray, bits_per_weight: int = 16):
        """
        Quantize float weights in [-8.0, 8.0] into fixed-point 16-bit integers,
        then unpack into a flat bit list for bit-wise Dual Regev fs-IBE encryption.
        """
        flat = np.concatenate([coef.flatten(), intercept.flatten()])
        clipped = np.clip(flat, -8.0, 8.0)
        scaled = np.round((clipped + 8.0) / 16.0 * ((1 << bits_per_weight) - 1)).astype(np.uint16)

        bits = []
        for val in scaled:
            for b in range(bits_per_weight - 1, -1, -1):
                bits.append(int((val >> b) & 1))
        return bits, len(flat)

    def encrypt_and_sign_update(self, target_server_id: str, epoch: int, bits_per_weight: int = 16):
        """
        1. Quantize local model weights (coef_ + intercept_) into bits.
        2. Encrypt each bit using Lattice fs-IBE (Dual Regev) for (target_server_id, epoch).
        3. Sign the serialized update metadata + ciphertext digest with CRYSTALS-Dilithium-3.
        """
        t0 = time.perf_counter()
        bits, num_weights = self._quantize_weights_to_bits(
            self.local_model.coef_, self.local_model.intercept_, bits_per_weight
        )

        # Encrypt bits under fs-IBE for the Central Aggregator at current epoch
        ciphertexts = [self.ops.Encrypt(target_server_id, epoch, b) for b in bits]
        enc_time = time.perf_counter() - t0

        # Construct message to sign with Dilithium-3
        t_sig0 = time.perf_counter()
        c2_bytes = bytes([ct["c2"] & 0xFF for ct in ciphertexts])
        msg_to_sign = (
            f"{self.gateway_id}|{epoch}|{self.num_samples}|".encode("utf-8") + c2_bytes
        )
        signature = self.signer.sign(msg_to_sign, self.sk_sig)
        sig_time = time.perf_counter() - t_sig0

        return {
            "gateway_id": self.gateway_id,
            "epoch": epoch,
            "num_samples": self.num_samples,
            "num_weights": num_weights,
            "bits_per_weight": bits_per_weight,
            "ciphertexts": ciphertexts,
            "signature": signature,
            "pk_sig": self.pk_sig,
            "msg_signed": msg_to_sign,
            "enc_time_s": enc_time,
            "sig_time_s": sig_time,
        }

    def evaluate_global(self, X_test: np.ndarray, y_test: np.ndarray):
        """Evaluate current model on test dataset."""
        X_scaled = self.scaler.transform(X_test)
        preds = self.local_model.predict(X_scaled)
        return {
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall": recall_score(y_test, preds, zero_division=0),
            "f1": f1_score(y_test, preds, zero_division=0),
        }