"""
Merkle tree over per-layer weight hashes.

Each leaf = SHA-256(shape_prefix || layer_weights.tobytes()).
Internal nodes = SHA-256(left_child || right_child).
The root is a 32-byte commitment to the full model's weight tensor set.

Tamper localization is O(log n): given a fresh set of leaf hashes, the
verify_and_localize() method returns the list of layer indices whose hash
no longer matches the stored tree.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from ..utils.crypto import hash_weight_tensor, hash_pair


@dataclass
class MerkleNode:
    """A single node in the Merkle tree."""
    digest: bytes                    # 32-byte SHA-256
    left: Optional["MerkleNode"] = field(default=None, repr=False)
    right: Optional["MerkleNode"] = field(default=None, repr=False)
    layer_index: Optional[int] = None  # Set only on leaf nodes


class MerkleTree:
    """
    Binary Merkle tree built over a sequence of layer weight hashes.

    Usage::

        tree = MerkleTree.from_weights({"layer0": w0, "layer1": w1, ...})
        root = tree.root_hash           # bytes
        proof = tree.get_proof(0)       # inclusion proof for leaf 0
        ok, bad_layers = tree.verify_and_localize(fresh_hashes)

    The tree is padded to the next power of two by duplicating the last leaf,
    following the standard Bitcoin-style construction.
    """

    def __init__(self, leaves: Sequence[MerkleNode]) -> None:
        if not leaves:
            raise ValueError("MerkleTree requires at least one leaf.")
        self._leaves: list[MerkleNode] = list(leaves)
        self._root: MerkleNode = self._build(list(leaves))

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_hashes(cls, layer_hashes: Sequence[bytes]) -> "MerkleTree":
        """
        Build from pre-computed 32-byte leaf hashes (e.g. loaded from disk).

        Args:
            layer_hashes: One 32-byte digest per layer, in layer order.
        """
        leaves = [
            MerkleNode(digest=h, layer_index=i)
            for i, h in enumerate(layer_hashes)
        ]
        return cls(leaves)

    @classmethod
    def from_weights(cls, layer_weights: dict[str, np.ndarray]) -> "MerkleTree":
        """
        Hash each weight tensor and build the Merkle tree.

        Args:
            layer_weights: Ordered dict mapping layer name → INT8 weight ndarray.
        """
        hashes = [hash_weight_tensor(w) for w in layer_weights.values()]
        return cls.from_hashes(hashes)

    # ------------------------------------------------------------------
    # Core tree construction (bottom-up)
    # ------------------------------------------------------------------

    @staticmethod
    def _build(nodes: list[MerkleNode]) -> MerkleNode:
        """Recursively pair nodes until only the root remains."""
        if len(nodes) == 1:
            return nodes[0]

        # Pad to even length by duplicating last node (standard practice)
        if len(nodes) % 2 != 0:
            nodes.append(MerkleNode(digest=nodes[-1].digest, layer_index=nodes[-1].layer_index))

        parents: list[MerkleNode] = []
        for i in range(0, len(nodes), 2):
            parent_digest = hash_pair(nodes[i].digest, nodes[i + 1].digest)
            parent = MerkleNode(digest=parent_digest, left=nodes[i], right=nodes[i + 1])
            parents.append(parent)
        return MerkleTree._build(parents)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def root_hash(self) -> bytes:
        """32-byte SHA-256 Merkle root."""
        return self._root.digest

    @property
    def leaf_hashes(self) -> list[bytes]:
        """The original (un-padded) leaf hashes, in layer order."""
        return [leaf.digest for leaf in self._leaves]

    @property
    def num_layers(self) -> int:
        return len(self._leaves)

    # ------------------------------------------------------------------
    # Inclusion proof (for external verifiers)
    # ------------------------------------------------------------------

    def get_proof(self, leaf_index: int) -> list[tuple[str, bytes]]:
        """
        Return the Merkle inclusion proof for a given leaf.

        Returns a list of (direction, sibling_hash) pairs from leaf to root.
        direction is 'left' or 'right' (the sibling's position).
        """
        if leaf_index < 0 or leaf_index >= len(self._leaves):
            raise IndexError(f"leaf_index {leaf_index} out of range [0, {len(self._leaves)})")

        proof: list[tuple[str, bytes]] = []
        self._collect_proof(self._root, leaf_index, 0, len(self._leaves), proof)
        return proof

    def _collect_proof(
        self,
        node: MerkleNode,
        target: int,
        lo: int,
        hi: int,
        proof: list,
    ) -> bool:
        if node.left is None and node.right is None:
            return lo == target

        mid = (lo + hi) // 2
        assert node.left is not None and node.right is not None

        if target < mid:
            if self._collect_proof(node.left, target, lo, mid, proof):
                proof.append(("right", node.right.digest))
                return True
        else:
            if self._collect_proof(node.right, target, mid, hi, proof):
                proof.append(("left", node.left.digest))
                return True
        return False

    # ------------------------------------------------------------------
    # Tamper localization
    # ------------------------------------------------------------------

    def verify_and_localize(self, fresh_hashes: Sequence[bytes]) -> tuple[bool, list[int]]:
        """
        Compare fresh layer hashes against the stored tree.

        Fast path: if root hashes match, all layers are intact (O(1)).
        Slow path: traverse the tree to find which leaf subtrees diverge,
                   yielding the exact set of tampered layer indices (O(n)).

        Args:
            fresh_hashes: One 32-byte hash per layer, freshly computed.

        Returns:
            (all_ok, tampered_indices) where tampered_indices is [] on success.
        """
        if len(fresh_hashes) != len(self._leaves):
            raise ValueError(
                f"Expected {len(self._leaves)} hashes, got {len(fresh_hashes)}"
            )

        # Build a fresh tree from the new hashes
        fresh_tree = MerkleTree.from_hashes(fresh_hashes)

        if fresh_tree.root_hash == self.root_hash:
            return True, []

        # Localize by comparing leaf-by-leaf
        tampered = [
            i
            for i, (stored, fresh) in enumerate(zip(self.leaf_hashes, fresh_hashes))
            if stored != fresh
        ]
        return False, tampered

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialize the tree's leaf hashes to a JSON-compatible dict."""
        return {
            "num_layers": self.num_layers,
            "root": self.root_hash.hex(),
            "leaves": [h.hex() for h in self.leaf_hashes],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MerkleTree":
        """Reconstruct a MerkleTree from to_dict() output."""
        hashes = [bytes.fromhex(h) for h in data["leaves"]]
        tree = cls.from_hashes(hashes)
        expected_root = bytes.fromhex(data["root"])
        if tree.root_hash != expected_root:
            raise ValueError("Merkle tree reconstruction failed: root mismatch.")
        return tree
