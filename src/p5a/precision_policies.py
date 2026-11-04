"""
Precision Policies for Dynamic MoE Layers
Includes:
  - ForcedSequencePolicy: Test-only deterministic sequence injector for G3 verification
  - FixedPrecisionPolicy: Static precision baseline
  - S4LegacyPolicy: Unconstrained joint cost optimization (diagnosed in D.1)
  - S4QualityBudgetPolicy: Constrained quality budget optimization (Phase D)
"""

from typing import Dict, Tuple, List, Optional
import pandas as pd
from src.tiered_cache import TieredCache

class ForcedSequencePolicy:
    """
    Test-only policy that forces a predetermined sequence of precisions
    for specific (layer_id, expert_id) pairs across consecutive calls.
    """
    def __init__(self, sequence_map: Dict[Tuple[int, int], List[str]], default_precision: str = "int4_g128"):
        self.sequence_map = {k: list(v) for k, v in sequence_map.items()}
        self.call_indices = {k: 0 for k in sequence_map}
        self.default_precision = default_precision

    def select(self, layer_id: int, expert_id: int, router_score: float = 0.0, cache: Optional[TieredCache] = None) -> str:
        k = (layer_id, expert_id)
        if k in self.sequence_map and len(self.sequence_map[k]) > 0:
            idx = self.call_indices[k]
            seq = self.sequence_map[k]
            prec = seq[idx % len(seq)]
            self.call_indices[k] += 1
            return prec
        return self.default_precision

class FixedPrecisionPolicy:
    def __init__(self, precision: str):
        self.precision = precision

    def select(self, layer_id: int, expert_id: int, router_score: float = 0.0, cache: Optional[TieredCache] = None) -> str:
        return self.precision

class S4LegacyPolicy:
    """
    Original unconstrained cost optimization formula from P4:
    Cost = lambda_quality * (score * error) + lambda_latency * load_cost
    """
    def __init__(self, sensitivity_lookup: Dict[Tuple[int, int], Dict[str, float]], lambda_q: float = 12.0, lambda_l: float = 1.0):
        self.sensitivity_lookup = sensitivity_lookup
        self.lambda_q = lambda_q
        self.lambda_l = lambda_l

    def get_error(self, layer_id: int, expert_id: int, prec: str) -> float:
        k = (layer_id, expert_id)
        if k in self.sensitivity_lookup:
            return self.sensitivity_lookup[k].get(prec, 0.20)
        return 0.0 if prec == "bf16" else (0.012 if prec.startswith("int8") else 0.18)

    def select(self, layer_id: int, expert_id: int, router_score: float, cache: Optional[TieredCache] = None) -> str:
        candidates = ["int4_g128", "int8_g128", "bf16"]
        best_p = "int4_g128"
        min_cost = float("inf")

        for cand in candidates:
            err = router_score * self.get_error(layer_id, expert_id, cand)
            load_cost = 0.0
            if cache is not None:
                k = cache._make_key(layer_id, expert_id, cand)
                if k in cache.vram_cache:
                    load_cost = 0.0
                elif k in cache.ram_cache:
                    load_cost = 0.2 if cand.startswith("int4") else (0.4 if cand.startswith("int8") else 0.8)
                else:
                    load_cost = 1.0 if cand.startswith("int4") else (2.0 if cand.startswith("int8") else 4.0)

            cost = self.lambda_q * err + self.lambda_l * load_cost
            if cost < min_cost:
                min_cost = cost
                best_p = cand
        return best_p
