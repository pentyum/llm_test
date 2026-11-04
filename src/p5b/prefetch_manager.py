"""
P5b Gate G5: Asynchronous Prefetch Manager
Implements:
  - sync_no_prefetch: Baseline, purely on-demand synchronous loading
  - static_layer_prefetch: Deployable strategy prefetching layer L+1 hot experts during layer L compute
  - oracle_prefetch: Upper-bound baseline using pre-recorded exact future router decisions
Includes strict tracking of prefetch hits, misses, and wasted bytes without memory leaks.
"""

import os
import sys
from typing import Dict, List, Optional, Any, Set
from collections import defaultdict

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.tiered_cache import TieredCache

class PrefetchManager:
    def __init__(
        self,
        cache: TieredCache,
        strategy: str = "sync_no_prefetch",
        target_layers: Optional[List[int]] = None,
        hot_experts_per_layer: Optional[Dict[int, List[int]]] = None,
        oracle_trace: Optional[List[Dict[str, Any]]] = None
    ):
        self.cache = cache
        self.strategy = strategy
        self.target_layers = sorted(list(target_layers or [0, 4, 8, 12]))
        # Default hot experts from Phase D activation distribution if not supplied
        self.hot_experts = hot_experts_per_layer or {
            0: [6, 10, 38, 47],
            4: [12, 23, 47, 55],
            8: [0, 11, 28, 45],
            12: [10, 24, 33, 50]
        }
        self.oracle_trace = oracle_trace or []
        self.oracle_index = 0

        self.prefetched_keys: Set[str] = set()
        self.hit_keys: Set[str] = set()
        self.prefetched_bytes = 0
        self.wasted_bytes = 0

    def on_expert_access(self, layer_id: int, expert_id: int, precision: str):
        """Called whenever an expert is actually accessed by the execution path."""
        k = f"{layer_id}:{expert_id}:{precision}"
        if k in self.prefetched_keys:
            self.hit_keys.add(k)

    def on_layer_enter(self, curr_layer_id: int, decode_step: int, current_precision: str = "int4_g128"):
        """Called upon entering a layer to trigger overlap prefetching for upcoming layers."""
        if self.strategy == "sync_no_prefetch":
            return

        if self.strategy == "static_layer_prefetch":
            # Find next target layer in pipeline
            if curr_layer_id in self.target_layers:
                curr_idx = self.target_layers.index(curr_layer_id)
                if curr_idx + 1 < len(self.target_layers):
                    next_layer = self.target_layers[curr_idx + 1]
                    # Asynchronously prefetch top hot experts for next layer to RAM
                    cands = self.hot_experts.get(next_layer, [0, 1])[:2]
                    for exp in cands:
                        k = f"{next_layer}:{exp}:{current_precision}"
                        if k not in self.prefetched_keys:
                            self.cache.prefetch(next_layer, exp, current_precision, target_tier="RAM")
                            self.prefetched_keys.add(k)
                            self.prefetched_bytes += 3200000 # Approx size of INT4 expert

        elif self.strategy == "oracle_prefetch":
            # Look ahead in oracle trace
            if self.oracle_index < len(self.oracle_trace):
                ahead_events = self.oracle_trace[self.oracle_index : self.oracle_index + 4]
                for ev in ahead_events:
                    l = ev["layer_id"]
                    e = ev["expert_id"]
                    p = ev.get("precision", current_precision)
                    k = f"{l}:{e}:{p}"
                    if k not in self.prefetched_keys:
                        self.cache.prefetch(l, e, p, target_tier="RAM")
                        self.prefetched_keys.add(k)
                        self.prefetched_bytes += 3200000
                self.oracle_index += 1

    def get_prefetch_stats(self) -> Dict[str, Any]:
        total_p = len(self.prefetched_keys)
        hits = len(self.hit_keys)
        wastes = total_p - hits
        wasted_bytes = wastes * 3200000
        hit_rate = (hits / total_p) if total_p > 0 else 0.0

        return {
            "strategy": self.strategy,
            "prefetch_requests": total_p,
            "prefetch_hits": hits,
            "prefetch_wastes": wastes,
            "prefetch_hit_rate": round(hit_rate, 4),
            "prefetched_bytes": self.prefetched_bytes,
            "wasted_bytes": wasted_bytes
        }
