import pytest
import os
import sys

def test_no_hardcoded_metrics_in_aggregate_script():
    """G7 Verification: aggregate_p5a3.py must derive all metrics from CSV/JSONL without hardcoded constants."""
    script_path = "/home/yaoyao/llm_test/scripts/aggregate_p5a3.py"
    assert os.path.exists(script_path)

    with open(script_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Disallow fake hardcoded token match rates or speeds in script logic
    assert "token_match_pct = 95.5" not in content
    assert "tpot_mean_ms = 77.0" not in content
    assert "df_gen.groupby" in content, "Must derive summary from df_gen"
    assert "df_tok.groupby" in content, "Must derive performance from df_tok"
