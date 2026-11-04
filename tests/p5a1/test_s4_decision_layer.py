import pytest
import os
import sys
import json
import pandas as pd

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.p5a.precision_policies import S4LegacyPolicy, S4QualityBudgetPolicy

REAL_TRACE_FILE = "/home/yaoyao/llm_test/artifacts/traces/trace_real_router.jsonl"
REAL_SENSITIVITY_FILE = "/home/yaoyao/llm_test/results/p45_real_sensitivity.csv"

@pytest.fixture
def sensitivity_lookup():
    df = pd.read_csv(REAL_SENSITIVITY_FILE)
    lookup = {}
    for _, row in df.iterrows():
        k = (int(row["layer_id"]), int(row["expert_id"]))
        lookup[k] = {
            "bf16": 0.0,
            "int8_g128": float(row["int8_rel_l2"]),
            "int4_g128": float(row["int4_rel_l2"])
        }
    return lookup

def test_s4_decision_layer_dual_precision_and_switching(sensitivity_lookup):
    """
    G5 Verification: Evaluates S4 on REAL router trace events from Phase D.
    Ensures S4_quality_budget produces BOTH INT4 and INT8 decisions and triggers switches.
    """
    assert os.path.exists(REAL_TRACE_FILE), "Real trace file missing!"

    # Budget threshold 0.020
    s4_budget = S4QualityBudgetPolicy(sensitivity_lookup, max_loss_budget=0.020)
    s4_legacy = S4LegacyPolicy(sensitivity_lookup, lambda_q=12.0, lambda_l=1.0)

    with open(REAL_TRACE_FILE, "r", encoding="utf-8") as f:
        events = [json.loads(line) for line in f if line.strip()][:500]

    budget_decisions = []
    legacy_decisions = []

    # Track decision history for each expert
    expert_history = {}

    for ev in events:
        l = ev["layer_id"]
        e = ev["expert_id"]
        score = ev["router_score"]

        p_budget = s4_budget.select(l, e, score)
        p_legacy = s4_legacy.select(l, e, score)

        budget_decisions.append(p_budget)
        legacy_decisions.append(p_legacy)

        if (l, e) not in expert_history:
            expert_history[(l, e)] = []
        expert_history[(l, e)].append((score, p_budget))

    # 1. Check S4_legacy vs S4_quality_budget
    # Legacy heavily favors INT4
    pct_int4_legacy = legacy_decisions.count("int4_g128") / len(legacy_decisions) * 100
    print(f"Legacy S4 INT4 percentage: {pct_int4_legacy:.1f}%")

    # Budget policy must have both INT4 and INT8
    count_int4 = budget_decisions.count("int4_g128")
    count_int8 = budget_decisions.count("int8_g128")
    print(f"S4 Quality Budget INT4: {count_int4}, INT8: {count_int8}")

    assert count_int4 > 0, "S4 must select INT4 for lower-score tokens"
    assert count_int8 > 0, "S4 must select INT8 for high-impact tokens"

    # 2. Check at least one expert experienced an autonomous switch due to router score change
    switched_experts = 0
    for (l, e), hist in expert_history.items():
        precs = set(p for _, p in hist)
        if len(precs) > 1:
            switched_experts += 1
            print(f"Expert ({l}, {e}) experienced autonomous switch: {[p for _, p in hist[:5]]}")

    assert switched_experts > 0, "At least one expert must switch precision autonomously on real router scores!"
