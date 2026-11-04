import pytest
import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.p5a.metrics import PrecisionEvent, compute_metrics_from_events

def test_prefill_only_precision_change_not_counted_as_decode_switch():
    """Prefill token_idx variations must NOT be counted as cross-decode token switching."""
    events = [
        # Prefill token 0 -> INT4
        PrecisionEvent(
            run_id="test", mode="dynamic_s4", phase="prefill",
            forward_id=1, generated_token_step=None, token_idx_in_forward=0,
            layer_id=0, expert_id=10, topk_rank=0, router_score=0.1,
            requested_precision="int4_g128", loaded_precision="int4_g128", executed_precision="int4_g128",
            hit_tier="VRAM", cache_key="0:10:int4_g128:1.0"
        ),
        # Prefill token 1 -> INT8 (same expert 10)
        PrecisionEvent(
            run_id="test", mode="dynamic_s4", phase="prefill",
            forward_id=1, generated_token_step=None, token_idx_in_forward=1,
            layer_id=0, expert_id=10, topk_rank=0, router_score=0.6,
            requested_precision="int8_g128", loaded_precision="int8_g128", executed_precision="int8_g128",
            hit_tier="VRAM", cache_key="0:10:int8_g128:1.0"
        )
    ]

    m = compute_metrics_from_events(events)
    # Must be 0 because both are in prefill phase!
    assert m["same_expert_decode_switch_count"] == 0
    assert m["switch_verified"] is False

def test_cross_decode_step_switching_detected():
    """Detects genuine precision switching across different decode generation steps."""
    events = [
        # Decode step 0: expert 10 executed as INT4
        PrecisionEvent(
            run_id="test", mode="dynamic_s4", phase="decode",
            forward_id=2, generated_token_step=0, token_idx_in_forward=0,
            layer_id=0, expert_id=10, topk_rank=0, router_score=0.1,
            requested_precision="int4_g128", loaded_precision="int4_g128", executed_precision="int4_g128",
            hit_tier="VRAM", cache_key="0:10:int4_g128:1.0"
        ),
        # Decode step 1: expert 10 executed as INT8
        PrecisionEvent(
            run_id="test", mode="dynamic_s4", phase="decode",
            forward_id=3, generated_token_step=1, token_idx_in_forward=0,
            layer_id=0, expert_id=10, topk_rank=0, router_score=0.7,
            requested_precision="int8_g128", loaded_precision="int8_g128", executed_precision="int8_g128",
            hit_tier="RAM", cache_key="0:10:int8_g128:1.0"
        ),
        # Decode step 2: expert 10 executed as INT4 again
        PrecisionEvent(
            run_id="test", mode="dynamic_s4", phase="decode",
            forward_id=4, generated_token_step=2, token_idx_in_forward=0,
            layer_id=0, expert_id=10, topk_rank=0, router_score=0.1,
            requested_precision="int4_g128", loaded_precision="int4_g128", executed_precision="int4_g128",
            hit_tier="VRAM", cache_key="0:10:int4_g128:1.0"
        )
    ]

    m = compute_metrics_from_events(events)
    assert m["same_expert_decode_switch_count"] == 2
    assert m["switch_verified"] is True
    assert m["switched_unique_experts_count"] == 1
    assert m["switch_transitions"]["int4_g128->int8_g128"] == 1
    assert m["switch_transitions"]["int8_g128->int4_g128"] == 1
    assert m["vram_target_hit_pct"] == 66.67
    assert m["ram_target_hit_pct"] == 33.33
