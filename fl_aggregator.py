"""
Federated Learning Aggregator Server (TA / PKG)
Verifies CRYSTALS-Dilithium-3 signatures from IoT Gateways, decrypts model updates
using epoch-bound Lattice fs-IBE secret keys, performs Federated Averaging (FedAvg),
and evolves keys forward across epochs (MinCov).
"""

import time
import numpy as np
from forward_security import UserOps
from Trust_Model import DilithiumSigner, TrustManager


class FederatedAggregatorServer:
    """
    Central FL Aggregator with Post-Quantum Forward-Secure Decryption.
    """

    def __init__(self, server_id: str, system: dict, signer: DilithiumSigner, n_features: int = 8):
        self.server_id = server_id
        self.system = system
        self.ops = UserOps(system)
        self.signer = signer
        self.trust_manager = TrustManager()
        self.n_features = n_features

        # Pre-generate epoch key bundle using Binary Tree Minimal Cover
        all_epochs = list(range(self.ops.total_epochs))
        self.key_bundle = self.ops.simulate_key_evolution(self.server_id, all_epochs)
        self.current_epoch = 1

        # Global model parameters
        self.global_coef = np.zeros((1, n_features), dtype=np.float64)
        self.global_intercept = np.zeros((1,), dtype=np.float64)

    @staticmethod
    def _dequantize_bits_to_weights(bits: list, n_features: int, bits_per_weight: int = 16):
        """Reconstruct float weights from decrypted bit array."""
        num_weights = n_features + 1
        vals = []
        for i in range(num_weights):
            chunk = bits[i * bits_per_weight : (i + 1) * bits_per_weight]
            int_val = sum(b << (bits_per_weight - 1 - j) for j, b in enumerate(chunk))
            float_val = (int_val / ((1 << bits_per_weight) - 1)) * 16.0 - 8.0
            vals.append(float_val)

        arr = np.array(vals, dtype=np.float64)
        coef = arr[:n_features].reshape(1, n_features)
        intercept = arr[n_features:].reshape(1,)
        return coef, intercept

    def receive_and_aggregate(self, encrypted_updates: list, epoch: int):
        """
        1. Check Gateway Trust Score & Verify Dilithium-3 Signature.
        2. Decrypt bit-level ciphertexts using server's epoch secret key sk_{server, epoch}.
        3. Aggregate decrypted weights using FedAvg (weighted by sample count).
        """
        valid_coefs = []
        valid_intercepts = []
        sample_weights = []

        total_verify_time = 0.0
        total_decrypt_time = 0.0
        accepted_gateways = 0
        rejected_gateways = 0

        for pkg in encrypted_updates:
            gw_id = pkg["gateway_id"]

            # 1. Trust check
            if not self.trust_manager.check(gw_id):
                rejected_gateways += 1
                continue

            # 2. Verify Dilithium-3 signature
            t_v0 = time.perf_counter()
            sig_ok = self.signer.verify(pkg["msg_signed"], pkg["signature"], pkg["pk_sig"])
            total_verify_time += time.perf_counter() - t_v0

            if not sig_ok or pkg["epoch"] != epoch:
                self.trust_manager.penalize(gw_id, amount=3.0)
                rejected_gateways += 1
                continue

            self.trust_manager.reward(gw_id, amount=1.0)

            # 3. Decrypt model weights using fs-IBE secret key for current epoch
            t_d0 = time.perf_counter()
            decrypted_bits = []
            decryption_failed = False
            for ct in pkg["ciphertexts"]:
                b = self.ops.Decrypt(ct, self.key_bundle)
                if b is None:
                    decryption_failed = True
                    break
                decrypted_bits.append(b)
            total_decrypt_time += time.perf_counter() - t_d0

            if decryption_failed:
                rejected_gateways += 1
                continue

            coef, intercept = self._dequantize_bits_to_weights(
                decrypted_bits, self.n_features, pkg["bits_per_weight"]
            )
            valid_coefs.append(coef)
            valid_intercepts.append(intercept)
            sample_weights.append(pkg["num_samples"])
            accepted_gateways += 1

        # 4. Federated Averaging (FedAvg): W_global = sum( (n_k / N) * W_k )
        if accepted_gateways > 0:
            total_samples = sum(sample_weights)
            self.global_coef = sum(
                c * (n_k / total_samples) for c, n_k in zip(valid_coefs, sample_weights)
            )
            self.global_intercept = sum(
                ic * (n_k / total_samples) for ic, n_k in zip(valid_intercepts, sample_weights)
            )

        return {
            "epoch": epoch,
            "accepted_gateways": accepted_gateways,
            "rejected_gateways": rejected_gateways,
            "verify_time_s": total_verify_time,
            "decrypt_time_s": total_decrypt_time,
            "global_coef_norm": float(np.linalg.norm(self.global_coef)),
        }

    def advance_epoch(self):
        """
        Evolve secret keys to t + 1 using Binary Tree Minimal Cover (MinCov)
        and permanently delete the old epoch key to guarantee Forward Security.
        """
        old_epoch = self.current_epoch
        _, min_cover_nodes = self.ops.Update(self.key_bundle, old_epoch)
        next_epoch = old_epoch + 1
        # Retain keys for future epochs (>= next_epoch) and delete all past epochs (<= old_epoch)
        self.key_bundle = {
            ep: sk for ep, sk in self.key_bundle.items() if ep >= next_epoch
        }
        self.current_epoch = next_epoch
        return old_epoch, self.current_epoch, min_cover_nodes