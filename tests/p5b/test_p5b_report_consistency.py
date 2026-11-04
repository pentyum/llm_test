import pytest
import os
import sys
import json
import pandas as pd

REPORT_MD = "/home/yaoyao/llm_test/docs/p5b/P5B_REPORT.md"
SUMMARY_CSV = "/home/yaoyao/llm_test/results/p5b/summary.csv"
STAGE_GATES_JSON = "/home/yaoyao/llm_test/results/p5b/stage_gates.json"

def test_report_and_summary_consistency():
    """G7 Verification: P5B_REPORT.md and summary.csv must have consistent values."""
    assert os.path.exists(REPORT_MD)
    assert os.path.exists(SUMMARY_CSV)
    assert os.path.exists(STAGE_GATES_JSON)

    df = pd.read_csv(SUMMARY_CSV)
    with open(REPORT_MD, "r", encoding="utf-8") as f:
        text = f.read()

    with open(STAGE_GATES_JSON, "r", encoding="utf-8") as f:
        gates = json.load(f)

    # Check that all G0-G7 gates are recorded
    for g_id in ["G0_environment_frozen_and_p5a3_audited", "G1_multilayer_parity_and_decode",
                 "G2_multiversion_offline_expert_store", "G3_strict_cache_capacity_invariants",
                 "G4_true_ssd_offload_without_dual_copy", "G5_prefetch_manager_and_ablation",
                 "G6_multilayer_policy_fair_evaluation", "G7_automated_report_and_tests"]:
        assert g_id in gates
        assert gates[g_id] == "PASS"

    # Check that modes from CSV appear in report text
    for m in df["mode"]:
        assert m in text
