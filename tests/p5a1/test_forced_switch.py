import pytest
import os
import sys
import torch

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.weight_store import WeightStore
from src.tiered_cache import TieredCache
from src.p5a.precision_policies import ForcedSequencePolicy
from src.p5a.metrics import PrecisionEvent, compute_metrics_from_events

REAL_EXPERTS_DIR = "/home/yaoyao/llm_test/artifacts/real_experts"

def test_forced_switch_int4_int8_int4_exactness():
    """
    G3 Verification: Deterministic forced sequence INT4 -> INT8 -> INT4 on expert (0, 12).
    Verifies that:
      1. Every step loads offline pre-quantized weights.
      2. Step 1 (INT4) and Step 3 (INT4) produce identical outputs on identical inputs.
      3. Step 2 (INT8) produces output numerically equivalent to direct INT8 loading.
      4. Transitions are recorded and verifiable.
    """
    store = WeightStore(root_dir=REAL_EXPERTS_DIR)
    cache = TieredCache(store)

    expert_id = 12
    sequence = ["int4_g128", "int8_g128", "int4_g128"]
    policy = ForcedSequencePolicy({(0, expert_id): sequence})

    torch.manual_seed(42)
    x = torch.randn(1, 2048, dtype=torch.bfloat16, device=cache.device)

    outputs = []
    events = []

    for step, expected_prec in enumerate(sequence):
        prec = policy.select(0, expert_id)
        assert prec == expected_prec

        gpu_data, meta, timings = cache.get_expert(0, expert_id, prec)
        assert meta["precision"] == expected_prec

        out, dec_ms, gemm_ms = cache.forward_expert_swiglu(x, gpu_data, meta)
        outputs.append(out.clone())

        ev = PrecisionEvent(
            run_id="test_forced",
            mode="forced",
            phase="decode",
            forward_id=step + 1,
            generated_token_step=step,
            token_idx_in_forward=0,
            layer_id=0,
            expert_id=expert_id,
            topk_rank=0,
            router_score=0.5,
            requested_precision=prec,
            loaded_precision=meta["precision"],
            executed_precision=meta["precision"],
            hit_tier=timings["hit_tier"],
            cache_key=f"0:{expert_id}:{prec}:1.0",
            total_ms=timings["total_ms"]
        )
        events.append(ev)

    # 1. Step 1 (INT4) and Step 3 (INT4) must be strictly identical
    assert torch.equal(outputs[0], outputs[2]), "INT4 repeated forward must be bitwise identical!"

    # 2. Step 2 (INT8) must differ from INT4 but be close
    rel_diff = (torch.norm(outputs[1].float() - outputs[0].float()) / torch.norm(outputs[1].float())).item()
    assert 0.05 < rel_diff < 0.30, f"INT8 vs INT4 relative difference unexpected: {rel_diff}"

    # 3. Direct load reference verification for INT8
    direct_data, direct_meta, _, _ = store.load_expert(0, expert_id, "int8_g128")
    direct_gpu = {k: v.to(cache.device) if isinstance(v, torch.Tensor) else v for k, v in direct_data.items()}
    ref_out, _, _ = cache.forward_expert_swiglu(x, direct_gpu, direct_meta)
    assert torch.equal(outputs[1], ref_out), "INT8 output must match direct offline load!"

    # 4. Metric verification
    m = compute_metrics_from_events(events)
    assert m["same_expert_decode_switch_count"] == 2
    assert m["switch_transitions"]["int4_g128->int8_g128"] == 1
    assert m["switch_transitions"]["int8_g128->int4_g128"] == 1
    assert m["switch_verified"] is True
