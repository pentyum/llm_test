import pytest
import os
from src.weight_store import WeightStore

def test_metadata_consistency_and_checksum():
    store = WeightStore()
    data, meta, _, file_bytes = store.load_expert(0, 0, "int4_g128", verify_checksum=True)

    assert meta["layer_id"] == 0
    assert meta["expert_id"] == 0
    assert meta["precision"] == "int4_g128"
    assert meta["group_size"] == 128
    assert "checksum" in meta
    assert meta["file_bytes"] == file_bytes

def test_nonexistent_expert_raises_error():
    store = WeightStore()
    with pytest.raises(FileNotFoundError):
        store.load_expert(99, 99, "int4_g128")
