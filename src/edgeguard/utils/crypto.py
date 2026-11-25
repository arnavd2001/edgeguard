"""
Cryptographic helpers: SHA-256 layer hashing, ECDSA sign/verify for Merkle roots.
"""

from __future__ import annotations

import hashlib
import struct
from typing import Sequence

import numpy as np

from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import (
    decode_dss_signature,
    encode_dss_signature,
)
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.backends import default_backend


# ---------------------------------------------------------------------------
# Layer weight hashing
# ---------------------------------------------------------------------------

def hash_weight_tensor(weights: np.ndarray) -> bytes:
    """
    Compute a SHA-256 digest over the raw bytes of a quantized weight tensor.

    For INT8 weights this is a deterministic 32-byte commitment to the full
    tensor.  Call this once per layer during model load; store as a Merkle leaf.

    Args:
        weights: numpy array of quantized weights (dtype int8, uint8, int32 …).

    Returns:
        32-byte SHA-256 digest.
    """
    h = hashlib.sha256()
    # Include shape as a prefix so that two tensors with the same values but
    # different shapes produce different hashes.
    shape_bytes = struct.pack(f">{len(weights.shape)}Q", *weights.shape)
    h.update(shape_bytes)
    h.update(weights.tobytes())
    return h.digest()


def hash_pair(left: bytes, right: bytes) -> bytes:
    """Combine two 32-byte digests into a parent node hash."""
    h = hashlib.sha256()
    h.update(left)
    h.update(right)
    return h.digest()


# ---------------------------------------------------------------------------
# ECDSA sign / verify (P-256)
# ---------------------------------------------------------------------------

def generate_keypair() -> tuple[ec.EllipticCurvePrivateKey, ec.EllipticCurvePublicKey]:
    """Generate a fresh P-256 keypair for signing Merkle roots."""
    private_key = ec.generate_private_key(ec.SECP256R1(), default_backend())
    return private_key, private_key.public_key()


def sign_root(root_hash: bytes, private_key: ec.EllipticCurvePrivateKey) -> bytes:
    """
    ECDSA-SHA256 signature over a Merkle root hash.

    Returns the DER-encoded signature bytes.
    """
    signature = private_key.sign(root_hash, ec.ECDSA(hashes.SHA256()))
    return signature


def verify_root(
    root_hash: bytes,
    signature: bytes,
    public_key: ec.EllipticCurvePublicKey,
) -> bool:
    """
    Verify an ECDSA-SHA256 signature over a Merkle root hash.

    Returns True if valid, False otherwise (never raises on bad signature).
    """
    try:
        public_key.verify(signature, root_hash, ec.ECDSA(hashes.SHA256()))
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Key serialization helpers
# ---------------------------------------------------------------------------

def private_key_to_pem(private_key: ec.EllipticCurvePrivateKey) -> bytes:
    return private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )


def public_key_to_pem(public_key: ec.EllipticCurvePublicKey) -> bytes:
    return public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )


def private_key_from_pem(pem: bytes) -> ec.EllipticCurvePrivateKey:
    return serialization.load_pem_private_key(pem, password=None, backend=default_backend())


def public_key_from_pem(pem: bytes) -> ec.EllipticCurvePublicKey:
    return serialization.load_pem_public_key(pem, backend=default_backend())
