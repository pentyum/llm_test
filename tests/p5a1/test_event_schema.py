import pytest
import os
import sys
import json
from dataclasses import asdict

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.p5a.metrics import PrecisionEvent, EventLogger

def test_event_schema_fields_and_json_nulls(tmp_path):
    jsonl_file = str(tmp_path / "test_events.jsonl")
    logger = EventLogger(jsonl_file)

    ev1 = PrecisionEvent(
        run_id="run_001",
        mode="dynamic_s4",
        phase="prefill",
        forward_id=1,
        generated_token_step=None,  # Must be None in prefill
        token_idx_in_forward=0,
        layer_id=0,
        expert_id=5,
        topk_rank=0,
        router_score=0.45,
        requested_precision="int4_g128",
        loaded_precision="int4_g128",
        executed_precision="int4_g128",
        hit_tier="VRAM",
        cache_key="0:5:int4_g128:1.0",
        fetch_ms=None,              # Unmeasured field
        dequant_ms=None,
        matmul_ms=None,
        total_ms=0.002
    )

    ev2 = PrecisionEvent(
        run_id="run_001",
        mode="dynamic_s4",
        phase="decode",
        forward_id=2,
        generated_token_step=0,     # Step 0 in decode
        token_idx_in_forward=0,
        layer_id=0,
        expert_id=5,
        topk_rank=1,
        router_score=0.25,
        requested_precision="int8_g128",
        loaded_precision="int4_g128",
        executed_precision="int4_g128",
        hit_tier="VRAM",
        cache_key="0:5:int4_g128:1.0",
        fetch_ms=0.002,
        dequant_ms=0.25,
        matmul_ms=0.05,
        total_ms=0.302,
        fallback_reason="int8_not_available_fallback_to_int4"
    )

    logger.log(ev1)
    logger.log(ev2)

    assert os.path.exists(jsonl_file)
    with open(jsonl_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    assert len(lines) == 2
    data1 = json.loads(lines[0])
    data2 = json.loads(lines[1])

    # Unmeasured fields should be serialized as null (None)
    assert data1["fetch_ms"] is None
    assert data1["generated_token_step"] is None
    assert data1["phase"] == "prefill"

    # Decode event verification
    assert data2["generated_token_step"] == 0
    assert data2["forward_id"] > data1["forward_id"]
    assert data2["fallback_reason"] is not None
    assert data2["requested_precision"] != data2["executed_precision"]
