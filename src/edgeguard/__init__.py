"""
EdgeGuard: Layerwise Intrusion Detection & Integrity Attestation
for Quantized DNNs on Edge Accelerators.

CS24M513 — IIT Madras MTech Project (Phase 1/2)
"""

__version__ = "0.1.0"
__author__  = "CS24M513"

from .attestation import MerkleAttestationManager
from .detection   import LayerwiseActivationMonitor
from .classifier  import ThreatClassifier, ThreatResult, ThreatType

__all__ = [
    "MerkleAttestationManager",
    "LayerwiseActivationMonitor",
    "ThreatClassifier",
    "ThreatResult",
    "ThreatType",
]
