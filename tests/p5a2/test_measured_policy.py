import pytest
import os
import sys
import pandas as pd

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.p5a.measured_cost_policy import S4MeasuredCostPolicy
from src.weight_store import WeightStore
from src.tiered_cache import TieredCache

REAL_SENSITIVITY_FILE = "/home/yaoyao/llm_test/results/p45_real_sensitivity.csv"
REAL_EXPERTS_DIR = "/home/yaoyao/llm_test/artifacts/real_experts"

@pytest.fixture
def sens_lookup():
    df = pd.read_csv(REAL_SENSITIVITY_FILE)
    lookup = {}
    for _, r in df.iterrows():
        lookup[(int(r["layer_id"]), int(r["expert_id"]))] = {
            "bf16": 0.0,
            "int8_g128": float(r["int8_rel_l2"]),
            "int4_g128": float(r["int4_rel_l2"])
        }
    return lookup

def test_quality_constraint_rejection(sens_lookup):
    """G7 Verification 1: Low-loss threshold rejects INT4 when router score is high."""
    policy = S4MeasuredCostPolicy(sens_lookup, max_loss_budget=0.010)

    # For expert (0, 0), INT4 error ~0.18. If score = 0.5, loss = 0.09 > 0.010
    chosen, debug = policy.select_with_debug(0, 0, router_score=0.5)

    assert "int4_g128" not in debug["valid_candidates"], "INT4 must be filtered out when loss exceeds budget!"
    assert chosen in ["int8_g128", "bf16"]

def test_cache_residency_cost_selection(sens_lookup):
    """G7 Verification 2: Prefers VRAM-resident precision when multiple precisions satisfy quality budget."""
    store = WeightStore(root_dir=REAL_EXPERTS_DIR)
    cache = TieredCache(store)

    policy = S4MeasuredCostPolicy(sens_lookup, max_loss_budget=0.050, enable_hysteresis=False)

    # Both INT4 and INT8 satisfy budget under small score 0.05 (INT4 loss ~0.009 <= 0.05)
    # Preload INT8 into VRAM
    cache.get_expert(0, 5, "int8_g128")

    # Select precision: INT8 in VRAM has ~0.2ms total cost, while INT4 from SSD has ~1.7ms
    chosen, debug = policy.select_with_debug(0, 5, router_score=0.05, cache=cache)

    assert "int4_g128" in debug["valid_candidates"]
    assert "int8_g128" in debug["valid_candidates"]
    assert chosen == "int8_g128", "Policy should select VRAM-resident INT8 over cold SSD INT4!"

def test_hysteresis_switch_damping(sens_lookup):
    """G7 Verification 3: Hysteresis prevents unnecessary precision thrashing when cost difference is marginal."""
    policy = S4MeasuredCostPolicy(sens_lookup, max_loss_budget=0.050, enable_hysteresis=True, hysteresis_ms=0.5)

    # Simulate step 1 chose INT4
    policy.last_precision_history[(0, 2)] = "int4_g128"

    # Step 2: Under similar cost, hysteresis favors staying in INT4
    chosen, debug = policy.select_with_debug(0, 2, router_score=0.05)
    assert chosen == "int4_g128"
