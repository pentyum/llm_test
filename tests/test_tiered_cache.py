import pytest
import torch
import time
from src.weight_store import WeightStore
from src.tiered_cache import TieredCache

def test_cache_keys_distinct_by_precision():
    store = WeightStore()
    cache = TieredCache(store)

    data_i4, meta_i4, _ = cache.get_expert(0, 0, "int4_g128")
    data_i8, meta_i8, _ = cache.get_expert(0, 0, "int8_g128")

    assert meta_i4["precision"] == "int4_g128"
    assert meta_i8["precision"] == "int8_g128"
    assert (0, 0, "int4_g128") in cache.vram_cache
    assert (0, 0, "int8_g128") in cache.vram_cache
    assert len(cache.vram_cache) >= 2

def test_cache_budgets_and_eviction():
    store = WeightStore()
    # Very small budget: 10 MB RAM, 7 MB VRAM (holds at most ~1-2 INT4/INT8 experts)
    cache = TieredCache(
        store,
        ram_budget_bytes=10 * 1024 * 1024,
        vram_budget_bytes=7 * 1024 * 1024,
        eviction_policy="lru"
    )

    # Load 4 different experts
    for e in range(4):
        cache.get_expert(0, e, "int4_g128")

    # Assert budget is strictly respected
    assert cache.vram_used <= 7 * 1024 * 1024
    assert cache.ram_used <= 10 * 1024 * 1024
    stats = cache.get_stats()
    assert stats["vram_evictions"] > 0
    assert stats["ram_evictions"] > 0

def test_no_race_in_prefetch():
    store = WeightStore()
    cache = TieredCache(store)

    # Prefetch expert (0, 5) to RAM
    cache.prefetch(0, 5, "int4_g128", target_tier="RAM")
    time.sleep(0.05) # Brief wait for async worker

    # Request same expert
    data, meta, timings = cache.get_expert(0, 5, "int4_g128")
    assert meta["expert_id"] == 5
    assert meta["precision"] == "int4_g128"
