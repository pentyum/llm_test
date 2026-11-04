import pytest
import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.p5a.measured_cost_policy import S4MeasuredCostPolicy
from src.p5a.precision_policies import ForcedSequencePolicy

def test_scheduler_no_online_quant(monkeypatch):
    import src.quant_codec
    def _forbidden(*args, **kwargs):
        raise RuntimeError("Online quant forbidden!")
    monkeypatch.setattr(src.quant_codec, "quantize_int4_g128", _forbidden)
    monkeypatch.setattr(src.quant_codec, "quantize_int8_g128", _forbidden)

    # Forced policy
    pol = ForcedSequencePolicy({(0, 0): ["int4_g128", "int8_g128"]})
    assert pol.select(0, 0) == "int4_g128"
    assert pol.select(0, 0) == "int8_g128"

def test_measured_cost_policy_select():
    pol = S4MeasuredCostPolicy({}, max_loss_budget=0.025)
    chosen = pol.select(0, 0, router_score=0.1)
    assert chosen in ["int4_g128", "int8_g128", "bf16"]
