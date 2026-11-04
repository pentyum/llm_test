"""
P5a.2 Measured Cost Policy (Gate G7)
Precision Scheduler using measured latencies from cost_profile.json
coupled with hard quality loss constraints:
  1. Filter candidates violating: router_score * expert_error <= max_loss_budget
  2. Among valid candidates, estimate realistic execution latency based on cache residency:
     T_est = T_fetch(precision, tier) + T_dequant(precision) + T_gemm(precision)
  3. Optional hysteresis penalty on precision switches to avoid thrashing.
  4. Select candidate with minimal expected latency.
"""

import os
import json
from typing import Dict, Tuple, List, Optional, Any
from src.tiered_cache import TieredCache

COST_PROFILE_JSON = "/home/yaoyao/llm_test/results/p5a2/cost_profile.json"

class S4MeasuredCostPolicy:
    def __init__(
        self,
        sensitivity_lookup: Dict[Tuple[int, int], Dict[str, float]],
        cost_profile_path: str = COST_PROFILE_JSON,
        max_loss_budget: float = 0.025,
        enable_hysteresis: bool = True,
        hysteresis_ms: float = 0.15,
        layer_defaults: Optional[Dict[str, float]] = None
    ):
        self.sensitivity_lookup = sensitivity_lookup
        self.max_loss_budget = max_loss_budget
        self.enable_hysteresis = enable_hysteresis
        self.hysteresis_ms = hysteresis_ms
        self.layer_defaults = layer_defaults or {
            "bf16": 0.0,
            "int8_g128": 0.012,
            "int4_g128": 0.180
        }

        # Load measured real latency costs
        self.cost_matrix = {}
        if os.path.exists(cost_profile_path):
            with open(cost_profile_path, "r", encoding="utf-8") as f:
                prof = json.load(f)
            metrics = prof.get("metrics", {})
            for p, data in metrics.items():
                self.cost_matrix[p] = {
                    "dequant_ms": data.get("dequant_ms", {}).get("mean", 0.3),
                    "gemm_ms": data.get("gemm_ms", {}).get("mean", 0.05),
                    "fetch_vram_ms": data.get("fetch_vram_ms", {}).get("mean", 0.001),
                    "fetch_ram_ms": data.get("fetch_ram_ms", {}).get("mean", 0.5),
                    "fetch_ssd_ms": data.get("fetch_ssd_ms", {}).get("mean", 1.5)
                }
        else:
            # Safe defaults if cost_profile.json ungenerated
            for p in ["int4_g128", "int8_g128", "bf16"]:
                self.cost_matrix[p] = {
                    "dequant_ms": 0.5 if "int4" in p else (0.2 if "int8" in p else 0.0),
                    "gemm_ms": 0.05,
                    "fetch_vram_ms": 0.001,
                    "fetch_ram_ms": 0.4 if "int4" in p else 0.7,
                    "fetch_ssd_ms": 1.2 if "int4" in p else 2.5
                }

        # Last executed precision history per expert: (layer_id, expert_id) -> precision
        self.last_precision_history: Dict[Tuple[int, int], str] = {}

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
        predicted_latencies = {}

        last_prec = self.last_precision_history.get((layer_id, expert_id))

        # 1. Evaluate hard quality loss constraint
        for cand in candidates:
            err = self.get_error(layer_id, expert_id, cand)
            loss = router_score * err
            losses[cand] = loss

            # Quality budget check (BF16 is always valid fallback)
            if loss <= self.max_loss_budget or cand == "bf16":
                valid_candidates.append(cand)

            # 2. Predict full execution latency T_est = T_fetch + T_dequant + T_gemm
            tier = "SSD"
            if cache is not None:
                k = cache._make_key(layer_id, expert_id, cand)
                if k in cache.vram_cache:
                    tier = "VRAM"
                elif k in cache.ram_cache:
                    tier = "RAM"

            c_info = self.cost_matrix.get(cand, {})
            t_fetch = c_info.get(f"fetch_{tier.lower()}_ms", 1.0)
            t_dequant = c_info.get("dequant_ms", 0.2)
            t_gemm = c_info.get("gemm_ms", 0.05)

            t_total = t_fetch + t_dequant + t_gemm

            # Apply hysteresis penalty if switching away from currently resident precision
            if self.enable_hysteresis and last_prec is not None and cand != last_prec:
                t_total += self.hysteresis_ms

            predicted_latencies[cand] = t_total

        if not valid_candidates:
            valid_candidates = ["bf16"]

        # 3. Select valid candidate with minimum predicted latency
        chosen = min(valid_candidates, key=lambda c: predicted_latencies[c])
        self.last_precision_history[(layer_id, expert_id)] = chosen

        debug = {
            "router_score": router_score,
            "losses": losses,
            "predicted_latencies": predicted_latencies,
            "valid_candidates": valid_candidates,
            "selected_precision": chosen
        }
        return chosen, debug

    def select(
        self,
        layer_id: int,
        expert_id: int,
        router_score: float,
        cache: Optional[TieredCache] = None
    ) -> str:
        chosen, _ = self.select_with_debug(layer_id, expert_id, router_score, cache)
        return chosen
