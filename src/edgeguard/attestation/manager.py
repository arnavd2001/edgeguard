"""
MerkleAttestationManager — high-level API for model weight attestation.

Workflow:
  1. build(layer_weights)       — hash all layers, build Merkle tree, sign root
  2. save(path)                 — persist tree + signature to disk
  3. load(path)                 — restore attestation state
  4. verify(layer_weights)      — fast-path root check; slow-path layer localization
  5. result.tampered_layers     — which layer indices were tampered ([] = clean)

This manager is intentionally framework-agnostic; it works with any dict mapping
layer name → numpy INT8 weight array, whether from PyTorch, TFLite, ONNX, or QNN.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .merkle import MerkleTree
from ..utils.crypto import (
    generate_keypair,
    sign_root,
    verify_root,
    hash_weight_tensor,
    private_key_to_pem,
    public_key_to_pem,
    private_key_from_pem,
    public_key_from_pem,
)


@dataclass
class AttestationResult:
    """Return value from MerkleAttestationManager.verify()."""
    # True iff the signed root is valid and all layer hashes match
    is_valid: bool
    # Indices of layers whose weight hash changed since attestation was built
    tampered_layers: list[int] = field(default_factory=list)
    # Human-readable names of tampered layers (populated if layer_names given)
    tampered_layer_names: list[str] = field(default_factory=list)
    # Whether the ECDSA signature over the Merkle root is valid
    signature_valid: bool = True
    # Timestamp (UTC epoch seconds) when the attestation was built
    attested_at: float = 0.0
    # Number of layers covered
    num_layers: int = 0

    @property
    def any_tampered(self) -> bool:
        return len(self.tampered_layers) > 0

    def summary(self) -> str:
        if self.is_valid:
            return f"CLEAN — {self.num_layers} layers verified, signature valid."
        parts = []
        if not self.signature_valid:
            parts.append("ROOT SIGNATURE INVALID")
        if self.tampered_layers:
            names = self.tampered_layer_names or [str(i) for i in self.tampered_layers]
            parts.append(f"TAMPERED LAYERS: {', '.join(names)}")
        return " | ".join(parts) if parts else "INVALID (unknown reason)"


class MerkleAttestationManager:
    """
    Manages the full lifecycle of Merkle-tree-based weight attestation.

    Attributes:
        tree:        The MerkleTree built from the model's weight hashes.
        layer_names: Ordered list of layer names, one per leaf.
        signature:   DER-encoded ECDSA signature over the Merkle root.
        attested_at: UTC epoch seconds when build() was called.
    """

    def __init__(self) -> None:
        self._tree: Optional[MerkleTree] = None
        self._layer_names: list[str] = []
        self._signature: Optional[bytes] = None
        self._public_key_pem: Optional[bytes] = None
        self._attested_at: float = 0.0

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(
        self,
        layer_weights: dict[str, np.ndarray],
        private_key=None,
    ) -> bytes:
        """
        Hash all layer weight tensors, build the Merkle tree, sign the root.

        Args:
            layer_weights: OrderedDict[layer_name → int8 ndarray].
            private_key:   An ECDSA P-256 private key (cryptography lib).
                           If None, a fresh throwaway key is generated.

        Returns:
            The 32-byte Merkle root hash.
        """
        if private_key is None:
            private_key, public_key = generate_keypair()
        else:
            public_key = private_key.public_key()

        self._layer_names = list(layer_weights.keys())
        self._tree = MerkleTree.from_weights(layer_weights)
        self._signature = sign_root(self._tree.root_hash, private_key)
        self._public_key_pem = public_key_to_pem(public_key)
        self._attested_at = time.time()

        return self._tree.root_hash

    # ------------------------------------------------------------------
    # Verify
    # ------------------------------------------------------------------

    def verify(self, layer_weights: dict[str, np.ndarray]) -> AttestationResult:
        """
        Verify the integrity of the model weights.

        1. ECDSA signature check on the stored root.
        2. Fresh per-layer hash → Merkle root recomputation.
        3. If roots match → CLEAN.  Else → localize tampered layers.

        Args:
            layer_weights: Current (possibly tampered) weight tensors.

        Returns:
            AttestationResult with full localization details.
        """
        self._require_built()

        result = AttestationResult(
            is_valid=False,
            attested_at=self._attested_at,
            num_layers=len(self._layer_names),
        )

        # Step 1: Verify ECDSA signature
        public_key = public_key_from_pem(self._public_key_pem)
        sig_ok = verify_root(self._tree.root_hash, self._signature, public_key)
        result.signature_valid = sig_ok
        if not sig_ok:
            return result

        # Step 2: Re-hash current weights
        if list(layer_weights.keys()) != self._layer_names:
            raise ValueError(
                "Layer names differ from attestation. "
                f"Expected: {self._layer_names}, Got: {list(layer_weights.keys())}"
            )
        fresh_hashes = [hash_weight_tensor(w) for w in layer_weights.values()]

        # Step 3: Merkle localization
        all_ok, tampered_indices = self._tree.verify_and_localize(fresh_hashes)
        result.tampered_layers = tampered_indices
        result.tampered_layer_names = [self._layer_names[i] for i in tampered_indices]
        result.is_valid = all_ok
        return result

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """
        Persist the attestation state to a JSON file.

        The file contains layer names, Merkle tree leaf hashes, the signed root,
        the ECDSA signature, and the public key (PEM).  The private key is NEVER
        stored.
        """
        self._require_built()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "version": 1,
            "attested_at": self._attested_at,
            "layer_names": self._layer_names,
            "tree": self._tree.to_dict(),
            "signature": self._signature.hex(),
            "public_key_pem": self._public_key_pem.decode("utf-8"),
        }
        path.write_text(json.dumps(state, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "MerkleAttestationManager":
        """Load an attestation previously saved with save()."""
        path = Path(path)
        state = json.loads(path.read_text())

        if state.get("version") != 1:
            raise ValueError(f"Unsupported attestation version: {state.get('version')}")

        mgr = cls()
        mgr._layer_names = state["layer_names"]
        mgr._tree = MerkleTree.from_dict(state["tree"])
        mgr._signature = bytes.fromhex(state["signature"])
        mgr._public_key_pem = state["public_key_pem"].encode("utf-8")
        mgr._attested_at = state["attested_at"]
        return mgr

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def root_hash(self) -> bytes:
        self._require_built()
        return self._tree.root_hash

    @property
    def layer_names(self) -> list[str]:
        return list(self._layer_names)

    @property
    def num_layers(self) -> int:
        return len(self._layer_names)

    def _require_built(self) -> None:
        if self._tree is None:
            raise RuntimeError("Call build() or load() before using the manager.")
