import pytest
import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.weight_store import WeightStore
from src.tiered_cache import TieredCache

REAL_EXPERTS_DIR = "/home/yaoyao/llm_test/artifacts/real_experts"

def test_independent_cache_isolation():
    """
    G7 Verification: Demonstrates that separate modes must use independent
    TieredCache instances, avoiding cross-strategy state pollution.
    """
    store = WeightStore(root_dir=REAL_EXPERTS_DIR)

    # 1. Mode 1 runs on Cache A
    cache_a = TieredCache(store, ram_budget_bytes=64*1024*1024, vram_budget_bytes=32*1024*1024)
    cache_a.get_expert(0, 1, "int4_g128")
    assert cache_a.vram_used > 0
    assert len(cache_a.vram_cache) == 1

    # 2. Mode 2 starts with fresh Cache B
    cache_b = TieredCache(store, ram_budget_bytes=64*1024*1024, vram_budget_bytes=32*1024*1024)
    assert cache_b.vram_used == 0
    assert cache_b.ram_used == 0
    assert len(cache_b.vram_cache) == 0
    assert len(cache_b.ram_cache) == 0

    # Cache B has no knowledge of Cache A
    stats_b = cache_b.get_stats()
    assert stats_b["requests"] == 0
    assert stats_b["vram_target_hits"] == 0
