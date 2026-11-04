"""
Precision Policies for Dynamic MoE Layers
Includes:
  - ForcedSequencePolicy: Test-only deterministic sequence injector for G3 verification
  - FixedPrecisionPolicy: Static precision baseline
  - S4LegacyPolicy: Unconstrained joint cost optimization (diagnosed in D.1)
  - S4QualityBudgetPolicy: Constrained quality budget optimization (Phase D, G5/G6)
"""

from typing import Dict, Tuple, List, Optional, Any
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
    Diagnosed: When load_cost difference is large, it easily degenerates to 100% INT4.
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

class S4QualityBudgetPolicy:
    """
    Constrained Quality Optimization Policy (Phase D):
    1. Filter out candidate precisions where estimated_loss = router_score * expert_error > max_loss_budget.
    2. Among valid candidates satisfying the quality budget, choose the lowest latency / memory cost.
    3. If INT4 satisfies budget (e.g. low router weight or robust expert), use INT4 for memory efficiency.
    4. If INT4 exceeds budget, upgrade to INT8 (or BF16) to protect high-impact tokens.
    """
    def __init__(
        self,
        sensitivity_lookup: Dict[Tuple[int, int], Dict[str, float]],
        max_loss_budget: float = 0.020,  # e.g. 0.020 allows low-weight experts to use INT4, forces high-weight to INT8
        layer_defaults: Optional[Dict[str, float]] = None
    ):
        self.sensitivity_lookup = sensitivity_lookup
        self.max_loss_budget = max_loss_budget
        self.layer_defaults = layer_defaults or {
            "bf16": 0.0,
            "int8_g128": 0.012,
            "int4_g128": 0.180
        }

    def get_error(self, layer_id: int, expert_id: int, prec: str) -> float:
        k = (layer_id, expert_id)
        if k in self.sensitivity_lookup:
            return self.sensitivity_lookup[k].get(prec, 0.20)
        return self.layer_defaults.get(prec, 0.20)

    def select_with_debug(
        self,
        layer_id: int,
        expert_id: int,
        router_score: float,
        cache: Optional[TieredCache] = None
    ) -> Tuple[str, Dict[str, Any]]:
        candidates = ["int4_g128", "int8_g128", "bf16"]
        valid_candidates = []
        losses = {}
        costs = {}

        for cand in candidates:
            err = self.get_error(layer_id, expert_id, cand)
            est_loss = router_score * err
            losses[cand] = est_loss

            # Practical load latency estimate in milliseconds
            load_cost = 0.0
            if cache is not None:
                k = cache._make_key(layer_id, expert_id, cand)
                if k in cache.vram_cache:
                    load_cost = 0.005 # VRAM hit
                elif k in cache.ram_cache:
                    load_cost = 0.26 if cand.startswith("int4") else (0.50 if cand.startswith("int8") else 0.96)
                else:
                    load_cost = 0.85 if cand.startswith("int4") else (1.50 if cand.startswith("int8") else 2.50)
            else:
                load_cost = 0.5 if cand.startswith("int4") else (1.0 if cand.startswith("int8") else 2.0)

            costs[cand] = load_cost
            if est_loss <= self.max_loss_budget or cand == "bf16":
                valid_candidates.append(cand)

        if not valid_candidates:
            valid_candidates = ["bf16"]

        # Among candidates satisfying quality budget, pick the one with minimal load cost
        best_cand = min(valid_candidates, key=lambda c: costs[c])
        debug_info = {
            "router_score": router_score,
            "losses": losses,
            "costs": costs,
            "valid_candidates": valid_candidates,
            "selected_precision": best_cand
        }
        return best_cand, debug_info

    def select(
        self,
        layer_id: int,
        expert_id: int,
        router_score: float,
        cache: Optional[TieredCache] = None
    ) -> str:
        chosen, _ = self.select_with_debug(layer_id, expert_id, router_score, cache)
        return chosen
