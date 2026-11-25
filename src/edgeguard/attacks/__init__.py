"""
EdgeGuard Attack Simulation Suite.

Implements realistic attack scenarios for evaluating detection robustness.
All attacks operate on numpy INT8 weight arrays — no real model required.

Attack classes:
  BitFlipAttack          — Rowhammer / BFA: targeted bit flips in weight tensors
  WeightExtractionAttack — Cold-boot / side-channel: read weights + dequantize
  TrainingDataRecovery   — Activation inversion: recover training data from weights
  SupplyChainPoisoning   — Backdoor injection into weight tensors
  AdversarialInputAttack — Input-space attacks on quantized models
  SynergyAttack          — Combined BFA + adversarial input
"""

from .bitflip import BitFlipAttack, BitFlipResult
from .weight_extraction import WeightExtractionAttack, ExtractionResult
from .training_data_recovery import TrainingDataRecovery, RecoveryResult
from .supply_chain import SupplyChainPoisoning, PoisoningResult
from .adversarial import AdversarialInputAttack, AdversarialResult
from .synergy import SynergyAttack, SynergyResult

__all__ = [
    "BitFlipAttack", "BitFlipResult",
    "WeightExtractionAttack", "ExtractionResult",
    "TrainingDataRecovery", "RecoveryResult",
    "SupplyChainPoisoning", "PoisoningResult",
    "AdversarialInputAttack", "AdversarialResult",
    "SynergyAttack", "SynergyResult",
]
