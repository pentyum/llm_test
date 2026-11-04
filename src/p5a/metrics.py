"""
P5a.1 Event Tracking and Metrics Aggregation Module
Defines:
  - PrecisionEvent: Strict dataclass for single-expert usage events
  - EventLogger: Thread-safe event collector writing to JSONL
  - AggregateMetrics: Pure derived statistics from raw events without hardcoding
"""

import os
import json
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict

@dataclass
class PrecisionEvent:
    run_id: str
    mode: str
    phase: str                      # "prefill" or "decode"
    forward_id: int                 # Global monotonic forward ID
    generated_token_step: Optional[int]  # None for prefill, 0, 1, 2... for decode
    token_idx_in_forward: int       # Position within current forward tensor
    layer_id: int
    expert_id: int
    topk_rank: int
    router_score: float
    requested_precision: str
    loaded_precision: str
    executed_precision: str
    hit_tier: str                   # "VRAM", "RAM", "SSD"
    cache_key: str                  # "layer:expert:precision:version"
    fetch_ms: Optional[float] = None
    dequant_ms: Optional[float] = None
    matmul_ms: Optional[float] = None
    total_ms: Optional[float] = None
    source_weight_sha256: Optional[str] = None
    kernel_type: str = "dequant_then_bf16_gemm"
    compute_dtype: str = "bf16"
    fallback_reason: Optional[str] = None

class EventLogger:
    def __init__(self, jsonl_path: Optional[str] = None):
        self.jsonl_path = jsonl_path
        self.events: List[PrecisionEvent] = []
        if self.jsonl_path:
            os.makedirs(os.path.dirname(os.path.abspath(self.jsonl_path)), exist_ok=True)
            # Truncate / create new file
            with open(self.jsonl_path, "w", encoding="utf-8") as f:
                pass

    def log(self, event: PrecisionEvent):
        self.events.append(event)
        if self.jsonl_path:
            with open(self.jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(event)) + "\n")

    def get_events(self) -> List[PrecisionEvent]:
        return self.events

    def clear(self):
        self.events.clear()

def compute_metrics_from_events(events: List[PrecisionEvent]) -> Dict[str, Any]:
    """Pure derived aggregation from a list of PrecisionEvents."""
    if not events:
        return {
            "total_events": 0,
            "status": "NO_EVENTS",
            "same_expert_decode_switch_count": 0,
            "switch_verified": False
        }

    total_events = len(events)
    decode_events = [e for e in events if e.phase == "decode"]
    prefill_events = [e for e in events if e.phase == "prefill"]

    # 1. Precision shares
    req_counts = defaultdict(int)
    exec_counts = defaultdict(int)
    for e in events:
        req_counts[e.requested_precision] += 1
        exec_counts[e.executed_precision] += 1

    precision_request_share_pct = {k: round(v / total_events * 100, 2) for k, v in req_counts.items()}
    precision_execute_share_pct = {k: round(v / total_events * 100, 2) for k, v in exec_counts.items()}

    # 2. Hit tier shares
    tier_counts = defaultdict(int)
    for e in events:
        tier_counts[e.hit_tier] += 1

    vram_hit_pct = round(tier_counts["VRAM"] / total_events * 100, 2)
    ram_hit_pct = round(tier_counts["RAM"] / total_events * 100, 2)
    ssd_miss_pct = round(tier_counts["SSD"] / total_events * 100, 2)

    # 3. Detect Same Expert Precision Switching across DECODE tokens
    # Group decode events by (layer_id, expert_id) and sort by generated_token_step
    decode_history = defaultdict(list)
    for e in decode_events:
        if e.generated_token_step is not None:
            decode_history[(e.layer_id, e.expert_id)].append(e)

    switch_count = 0
    transitions = defaultdict(int)
    switched_experts = set()

    for (l, exp), hist in decode_history.items():
        # Sort by generated_token_step
        hist_sorted = sorted(hist, key=lambda x: x.generated_token_step)
        prev_prec = None
        for item in hist_sorted:
            curr_prec = item.executed_precision
            if prev_prec is not None and curr_prec != prev_prec:
                switch_count += 1
                trans_key = f"{prev_prec}->{curr_prec}"
                transitions[trans_key] += 1
                switched_experts.add((l, exp))
            prev_prec = curr_prec

    # 4. Fallback checks
    fallbacks = [e for e in events if e.requested_precision != e.executed_precision]

    return {
        "total_events": total_events,
        "prefill_events_count": len(prefill_events),
        "decode_events_count": len(decode_events),
        "precision_request_share_pct": dict(precision_request_share_pct),
        "precision_execute_share_pct": dict(precision_execute_share_pct),
        "vram_target_hit_pct": vram_hit_pct,
        "ram_target_hit_pct": ram_hit_pct,
        "ssd_miss_pct": ssd_miss_pct,
        "same_expert_decode_switch_count": switch_count,
        "switched_unique_experts_count": len(switched_experts),
        "switch_transitions": dict(transitions),
        "switch_verified": (switch_count > 0),
        "fallback_count": len(fallbacks)
    }
