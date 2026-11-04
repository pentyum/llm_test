import pytest
import os
import sys
import pandas as pd

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

TEACHER_CSV = "/home/yaoyao/llm_test/results/p5a3/quality/teacher_forced_logits.csv"

def test_teacher_forced_metrics():
    assert os.path.exists(TEACHER_CSV), "teacher_forced_logits.csv must exist"
    df = pd.read_csv(TEACHER_CSV)

    # 32 prompts x 16 steps x 4 modes = 2048 rows
    assert len(df) == 32 * 16 * 4, f"Expected 2048 rows, got {len(df)}"
    assert len(df["prompt_id"].unique()) == 32

    # High top-1 agreement and low KL divergence across all modes
    mode_summary = df.groupby("mode").agg({
        "top1_agreement": "mean",
        "kl_divergence": "mean"
    })

    for m, row in mode_summary.iterrows():
        assert row["top1_agreement"] > 0.95, f"Mode {m} top-1 agreement too low: {row['top1_agreement']}"
        assert row["kl_divergence"] < 0.01, f"Mode {m} mean KL divergence too high: {row['kl_divergence']}"
