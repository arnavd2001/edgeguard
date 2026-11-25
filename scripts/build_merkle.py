#!/usr/bin/env python3
"""
Build and save a Merkle attestation for a quantized model.

Usage:
    # Generate a keypair and attest a model from a .npz weight file:
    python scripts/build_merkle.py --weights model_int8.npz --out attestation/

    # Regenerate keypair only:
    python scripts/build_merkle.py --keygen --out keys/

    # Verify a saved attestation against current weights:
    python scripts/build_merkle.py --verify --weights model_int8.npz \
        --attestation attestation/model_attestation.json
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Allow running from project root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from edgeguard.attestation.manager import MerkleAttestationManager
from edgeguard.utils.crypto import (
    generate_keypair,
    private_key_to_pem,
    public_key_to_pem,
    private_key_from_pem,
)


def load_weights(path: Path) -> dict[str, np.ndarray]:
    """Load INT8 weights from a .npz file."""
    data = np.load(path, allow_pickle=False)
    weights = {k: data[k] for k in data.files}
    print(f"[build_merkle] Loaded {len(weights)} layers from {path}")
    for name, w in weights.items():
        print(f"  {name}: shape={w.shape}, dtype={w.dtype}")
    return weights


def cmd_keygen(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    private_key, public_key = generate_keypair()
    priv_path = out / "edgeguard_private.pem"
    pub_path  = out / "edgeguard_public.pem"
    priv_path.write_bytes(private_key_to_pem(private_key))
    pub_path.write_bytes(public_key_to_pem(public_key))
    print(f"[build_merkle] Private key → {priv_path}")
    print(f"[build_merkle] Public key  → {pub_path}")
    print("[build_merkle] Keep the private key secure; share only the public key.")


def cmd_build(args):
    weights = load_weights(Path(args.weights))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    private_key = None
    if args.private_key:
        pem = Path(args.private_key).read_bytes()
        private_key = private_key_from_pem(pem)
        print(f"[build_merkle] Using private key from {args.private_key}")

    mgr = MerkleAttestationManager()
    root_hash = mgr.build(weights, private_key=private_key)
    print(f"[build_merkle] Merkle root: {root_hash.hex()}")
    print(f"[build_merkle] Layers: {mgr.num_layers}")

    attest_path = out / "model_attestation.json"
    mgr.save(attest_path)
    print(f"[build_merkle] Attestation saved → {attest_path}")


def cmd_verify(args):
    weights = load_weights(Path(args.weights))
    attest_path = Path(args.attestation)
    mgr = MerkleAttestationManager.load(attest_path)
    print(f"[build_merkle] Loaded attestation from {attest_path}")
    result = mgr.verify(weights)
    print(f"[build_merkle] {result.summary()}")
    if not result.is_valid:
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="EdgeGuard Merkle attestation builder")
    parser.add_argument("--weights",      help="Path to .npz INT8 weight file")
    parser.add_argument("--out",          default=".", help="Output directory")
    parser.add_argument("--private-key",  help="Path to PEM private key (optional; generated if absent)")
    parser.add_argument("--attestation",  help="Path to saved attestation JSON (for --verify)")
    parser.add_argument("--keygen",       action="store_true", help="Generate a new P-256 keypair")
    parser.add_argument("--verify",       action="store_true", help="Verify attestation against weights")
    args = parser.parse_args()

    if args.keygen:
        cmd_keygen(args)
    elif args.verify:
        if not args.weights or not args.attestation:
            parser.error("--verify requires --weights and --attestation")
        cmd_verify(args)
    else:
        if not args.weights:
            parser.error("--weights is required unless --keygen or --verify is specified")
        cmd_build(args)


if __name__ == "__main__":
    main()
