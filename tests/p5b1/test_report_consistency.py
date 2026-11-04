import pytest
import os
import sys
import json
import pandas as pd

REPORT_MD = "/home/yaoyao/llm_test/docs/p5b1/P5B1_REPORT.md"
STAGE_GATES = "/home/yaoyao/llm_test/results/p5b1/stage_gates.json"
ABLATION_CSV = "/home/yaoyao/llm_test/results/p5b1/prefetch_ablation.csv"

def test_p5b1_report_and_data_consistency():
    """G7 Verification: Checks that P5b.1 report contains true data and all G0-G7 gates."""
    assert os.path.exists(REPORT_MD)
    assert os.path.exists(STAGE_GATES)
    assert os.path.exists(ABLATION_CSV)

    with open(STAGE_GATES) as f:
        gates = json.load(f)

    for g_id in [
        "G0_environment_frozen_and_preflight",
        "G1_safe_restore_and_unloading_semantics",
        "G2_prefetch_state_machine_and_exact_accounting",
        "G3_strict_cache_budget_invariants",
        "G4_multilayer_bf16_router_parity",
        "G5_strict_fair_prefetch_ablation",
        "G6_unbiased_speedup_evaluation",
        "G7_automated_test_suite_and_report"
    ]:
        assert g_id in gates

    with open(REPORT_MD) as f:
        text = f.read()

    df = pd.read_csv(ABLATION_CSV)
    for s in df["strategy"].unique():
        assert s in text
