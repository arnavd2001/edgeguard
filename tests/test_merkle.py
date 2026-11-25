"""Tests for the Merkle tree attestation subsystem."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from edgeguard.attestation.merkle import MerkleTree, MerkleNode
from edgeguard.attestation.manager import MerkleAttestationManager, AttestationResult
from edgeguard.utils.crypto import hash_weight_tensor, generate_keypair


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_weights(n_layers: int = 4, seed: int = 42) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        f"layer{i}": rng.integers(-128, 128, size=(64, 64), dtype=np.int8)
        for i in range(n_layers)
    }


# ---------------------------------------------------------------------------
# MerkleTree unit tests
# ---------------------------------------------------------------------------

class TestMerkleTree:

    def test_single_layer(self):
        weights = make_weights(1)
        hashes = [hash_weight_tensor(w) for w in weights.values()]
        tree = MerkleTree.from_hashes(hashes)
        assert len(tree.root_hash) == 32
        assert tree.num_layers == 1

    def test_power_of_two_layers(self):
        for n in [2, 4, 8]:
            weights = make_weights(n)
            tree = MerkleTree.from_weights(weights)
            assert tree.num_layers == n

    def test_non_power_of_two(self):
        # 3, 5, 7 layers (odd / non-power-of-two)
        for n in [3, 5, 7]:
            weights = make_weights(n)
            tree = MerkleTree.from_weights(weights)
            assert tree.num_layers == n

    def test_deterministic_root(self):
        weights = make_weights(4)
        t1 = MerkleTree.from_weights(weights)
        t2 = MerkleTree.from_weights(weights)
        assert t1.root_hash == t2.root_hash

    def test_different_weights_different_root(self):
        t1 = MerkleTree.from_weights(make_weights(4, seed=1))
        t2 = MerkleTree.from_weights(make_weights(4, seed=2))
        assert t1.root_hash != t2.root_hash

    def test_verify_clean(self):
        weights = make_weights(4)
        tree = MerkleTree.from_weights(weights)
        fresh = [hash_weight_tensor(w) for w in weights.values()]
        ok, bad = tree.verify_and_localize(fresh)
        assert ok is True
        assert bad == []

    def test_verify_single_layer_tamper(self):
        weights = make_weights(4)
        tree = MerkleTree.from_weights(weights)

        # Tamper layer 2
        tampered = dict(weights)
        rng = np.random.default_rng(99)
        tampered["layer2"] = rng.integers(-128, 128, size=(64, 64), dtype=np.int8)

        fresh = [hash_weight_tensor(w) for w in tampered.values()]
        ok, bad = tree.verify_and_localize(fresh)
        assert ok is False
        assert 2 in bad
        assert 0 not in bad
        assert 1 not in bad

    def test_verify_multiple_layers_tampered(self):
        weights = make_weights(6)
        tree = MerkleTree.from_weights(weights)
        tampered = dict(weights)
        rng = np.random.default_rng(77)
        for idx in [0, 3, 5]:
            tampered[f"layer{idx}"] = rng.integers(-128, 128, size=(64, 64), dtype=np.int8)
        fresh = [hash_weight_tensor(w) for w in tampered.values()]
        ok, bad = tree.verify_and_localize(fresh)
        assert ok is False
        assert set(bad) == {0, 3, 5}

    def test_serialization_roundtrip(self):
        weights = make_weights(4)
        tree = MerkleTree.from_weights(weights)
        d = tree.to_dict()
        restored = MerkleTree.from_dict(d)
        assert restored.root_hash == tree.root_hash
        assert restored.leaf_hashes == tree.leaf_hashes


# ---------------------------------------------------------------------------
# MerkleAttestationManager integration tests
# ---------------------------------------------------------------------------

class TestMerkleAttestationManager:

    def test_build_and_verify_clean(self):
        weights = make_weights(4)
        mgr = MerkleAttestationManager()
        mgr.build(weights)
        result = mgr.verify(weights)
        assert result.is_valid is True
        assert result.tampered_layers == []
        assert result.signature_valid is True

    def test_build_with_custom_key(self):
        weights = make_weights(4)
        private_key, _ = generate_keypair()
        mgr = MerkleAttestationManager()
        mgr.build(weights, private_key=private_key)
        result = mgr.verify(weights)
        assert result.is_valid is True

    def test_detects_weight_tamper(self):
        weights = make_weights(4)
        mgr = MerkleAttestationManager()
        mgr.build(weights)

        tampered = dict(weights)
        rng = np.random.default_rng(55)
        tampered["layer1"] = rng.integers(-128, 128, size=(64, 64), dtype=np.int8)

        result = mgr.verify(tampered)
        assert result.is_valid is False
        assert 1 in result.tampered_layers
        assert "layer1" in result.tampered_layer_names

    def test_save_and_load(self):
        weights = make_weights(4)
        mgr = MerkleAttestationManager()
        mgr.build(weights)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "attestation.json"
            mgr.save(path)
            assert path.exists()

            mgr2 = MerkleAttestationManager.load(path)
            result = mgr2.verify(weights)
            assert result.is_valid is True

    def test_save_load_detects_tamper(self):
        weights = make_weights(4)
        mgr = MerkleAttestationManager()
        mgr.build(weights)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "attestation.json"
            mgr.save(path)
            mgr2 = MerkleAttestationManager.load(path)

        tampered = dict(weights)
        rng = np.random.default_rng(33)
        tampered["layer3"] = rng.integers(-128, 128, size=(64, 64), dtype=np.int8)
        result = mgr2.verify(tampered)
        assert result.is_valid is False
        assert 3 in result.tampered_layers

    def test_require_built_raises(self):
        mgr = MerkleAttestationManager()
        with pytest.raises(RuntimeError, match="build"):
            _ = mgr.root_hash

    def test_summary_clean(self):
        weights = make_weights(4)
        mgr = MerkleAttestationManager()
        mgr.build(weights)
        result = mgr.verify(weights)
        assert "CLEAN" in result.summary()

    def test_summary_tampered(self):
        weights = make_weights(4)
        mgr = MerkleAttestationManager()
        mgr.build(weights)
        tampered = dict(weights)
        rng = np.random.default_rng(11)
        tampered["layer0"] = rng.integers(-128, 128, size=(64, 64), dtype=np.int8)
        result = mgr.verify(tampered)
        assert "TAMPERED" in result.summary()
