"""
Tiered Cache for MoE Multi-Precision Experts
Tiers:
  - SSD: Persistent storage (read-only, no on-the-fly quant)
  - RAM: Host memory cache (configurable budget)
  - VRAM: GPU memory cache (configurable budget)
Policies:
  - LRU
  - Cost-Aware (reload_penalty / size)
"""

import os
import sys
import time
import threading
from typing import Dict, Tuple, Optional, Any, List
from collections import OrderedDict

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.weight_store import WeightStore
from src.quant_codec import dequantize_int8_g128, dequantize_int4_g128

class TieredCache:
    def __init__(
        self,
        weight_store: WeightStore,
        ram_budget_bytes: int = 2 * 1024 * 1024 * 1024,   # 2 GiB default
        vram_budget_bytes: int = 512 * 1024 * 1024,       # 512 MiB default
        eviction_policy: str = "lru",                     # "lru" or "cost_aware"
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.store = weight_store
        self.ram_budget = ram_budget_bytes
        self.vram_budget = vram_budget_bytes
        self.policy = eviction_policy
        self.device = device

        self.ram_used = 0
        self.vram_used = 0

        # Cache key: (layer_id, expert_id, precision)
        # Entry in RAM: (data_dict, meta_dict, size_bytes, last_accessed_time, access_count)
        self.ram_cache: Dict[Tuple[int, int, str], Tuple[Dict[str, Any], Dict[str, Any], int, float, int]] = OrderedDict()
        
        # Entry in VRAM: (vram_data_dict, meta_dict, size_bytes, last_accessed_time, access_count)
        self.vram_cache: Dict[Tuple[int, int, str], Tuple[Dict[str, Any], Dict[str, Any], int, float, int]] = OrderedDict()

        # Prefetch queue & tracking
        self.prefetch_threads: List[threading.Thread] = []
        self.lock = threading.Lock()

        # Detailed metrics
        self.stats = {
            "requests": 0,
            "vram_hits": 0,
            "ram_hits": 0,
            "ssd_reads": 0,
            "target_precision_hits": 0,
            "expert_any_precision_hits": 0,
            "ram_evictions": 0,
            "vram_evictions": 0,
            "ssd_read_bytes": 0,
            "ram_to_vram_bytes": 0,
            "total_io_ms": 0.0,
            "total_copy_ms": 0.0,
            "total_decode_ms": 0.0,
            "total_gemm_ms": 0.0,
            "prefetch_requests": 0,
            "prefetch_hits": 0
        }

    def _estimate_size_bytes(self, meta: Dict[str, Any], precision: str) -> int:
        if "file_bytes" in meta and meta["file_bytes"] > 0:
            return meta["file_bytes"]
        # Fallback to theoretical sizes for OLMoE (6.29M params)
        if precision == "bf16":
            return int(6291456 * 2)
        elif precision.startswith("int8"):
            return int(6291456 * 1 + 6291456 / 128 * 2)
        elif precision.startswith("int4"):
            return int(6291456 * 0.5 + 6291456 / 128 * 2)
        return 12 * 1024 * 1024

    def _evict(self, tier: str, needed_bytes: int):
        target_cache = self.vram_cache if tier == "vram" else self.ram_cache
        used = self.vram_used if tier == "vram" else self.ram_used
        budget = self.vram_budget if tier == "vram" else self.ram_budget

        while used + needed_bytes > budget and target_cache:
            if self.policy == "lru":
                evict_key, entry = target_cache.popitem(last=False)
            elif self.policy == "cost_aware":
                # Cost-aware priority: cost = (reload_penalty * access_count) / size
                # Higher cost = more valuable; we evict the lowest cost
                now = time.time()
                min_key = None
                min_score = float("inf")
                for k, v in target_cache.items():
                    size = v[2]
                    count = v[4]
                    recency = max(now - v[3], 0.001)
                    score = (count / recency) / max(size, 1024)
                    if score < min_score:
                        min_score = score
                        min_key = k
                evict_key = min_key
                entry = target_cache.pop(evict_key)
            else:
                evict_key, entry = target_cache.popitem(last=False)

            entry_bytes = entry[2]
            used -= entry_bytes
            if tier == "vram":
                self.stats["vram_evictions"] += 1
            else:
                self.stats["ram_evictions"] += 1

        if tier == "vram":
            self.vram_used = used
        else:
            self.ram_used = used

    def prefetch(self, layer_id: int, expert_id: int, precision: str, target_tier: str = "RAM"):
        """
        Background prefetch into RAM or VRAM without blocking.
        """
        key = (layer_id, expert_id, precision)
        with self.lock:
            self.stats["prefetch_requests"] += 1
            if target_tier == "VRAM" and key in self.vram_cache:
                return
            if target_tier == "RAM" and (key in self.ram_cache or key in self.vram_cache):
                return

        def _bg_fetch():
            try:
                data, meta, _, file_bytes = self.store.load_expert(layer_id, expert_id, precision)
                with self.lock:
                    if target_tier == "RAM":
                        self._evict("ram", file_bytes)
                        self.ram_cache[key] = (data, meta, file_bytes, time.time(), 1)
                        self.ram_used += file_bytes
                    elif target_tier == "VRAM" and self.device == "cuda":
                        self._evict("vram", file_bytes)
                        gpu_data = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in data.items()}
                        self.vram_cache[key] = (gpu_data, meta, file_bytes, time.time(), 1)
                        self.vram_used += file_bytes
            except Exception:
                pass

        t = threading.Thread(target=_bg_fetch, daemon=True)
        t.start()
        self.prefetch_threads.append(t)

    def get_expert(
        self,
        layer_id: int,
        expert_id: int,
        target_precision: str,
        allow_fallback: bool = False
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, float]]:
        """
        Fetches expert from VRAM -> RAM -> SSD.
        Returns: (expert_tensors_on_device, meta_dict, timing_breakdown)
        """
        timings = {
            "ssd_read_ms": 0.0,
            "h2d_copy_ms": 0.0,
            "decode_ms": 0.0,
            "gemm_ms": 0.0,
            "total_ms": 0.0
        }
        t_start = time.perf_counter()

        with self.lock:
            self.stats["requests"] += 1
            key = (layer_id, expert_id, target_precision)
            actual_precision = target_precision

            # Check any-precision expert hit
            any_hit = any(k[0] == layer_id and k[1] == expert_id for k in self.vram_cache) or \
                      any(k[0] == layer_id and k[1] == expert_id for k in self.ram_cache)
            if any_hit:
                self.stats["expert_any_precision_hits"] += 1

            # 1. Check VRAM cache hit
            if key in self.vram_cache:
                entry = self.vram_cache[key]
                self.vram_cache.move_to_end(key)
                self.vram_cache[key] = (entry[0], entry[1], entry[2], time.time(), entry[4] + 1)
                self.stats["vram_hits"] += 1
                self.stats["target_precision_hits"] += 1
                timings["total_ms"] = (time.perf_counter() - t_start) * 1000.0
                return entry[0], entry[1], timings

            # 2. Check RAM cache hit
            data_cpu = None
            meta = None
            size_bytes = 0

            if key in self.ram_cache:
                entry = self.ram_cache[key]
                self.ram_cache.move_to_end(key)
                self.ram_cache[key] = (entry[0], entry[1], entry[2], time.time(), entry[4] + 1)
                data_cpu, meta, size_bytes = entry[0], entry[1], entry[2]
                self.stats["ram_hits"] += 1
                self.stats["target_precision_hits"] += 1
            else:
                # 3. SSD Cold Read
                t_io_0 = time.perf_counter()
                data_cpu, meta, io_ms, size_bytes = self.store.load_expert(layer_id, expert_id, target_precision)
                t_io_1 = time.perf_counter()
                timings["ssd_read_ms"] = (t_io_1 - t_io_0) * 1000.0
                self.stats["ssd_reads"] += 1
                self.stats["ssd_read_bytes"] += size_bytes
                self.stats["total_io_ms"] += timings["ssd_read_ms"]

                # Insert into RAM cache
                self._evict("ram", size_bytes)
                self.ram_cache[key] = (data_cpu, meta, size_bytes, time.time(), 1)
                self.ram_used += size_bytes

            # Copy RAM -> VRAM
            t_copy_0 = time.perf_counter()
            gpu_data = {}
            for k, v in data_cpu.items():
                if isinstance(v, torch.Tensor):
                    gpu_data[k] = v.to(self.device, non_blocking=False)
                else:
                    gpu_data[k] = v

            if self.device == "cuda":
                torch.cuda.synchronize()
            t_copy_1 = time.perf_counter()
            timings["h2d_copy_ms"] = (t_copy_1 - t_copy_0) * 1000.0
            self.stats["ram_to_vram_bytes"] += size_bytes
            self.stats["total_copy_ms"] += timings["h2d_copy_ms"]

            # Insert into VRAM cache
            self._evict("vram", size_bytes)
            self.vram_cache[key] = (gpu_data, meta, size_bytes, time.time(), 1)
            self.vram_used += size_bytes

        timings["total_ms"] = (time.perf_counter() - t_start) * 1000.0
        return gpu_data, meta, timings

    def forward_expert_swiglu(self, x: torch.Tensor, gpu_data: Dict[str, Any], meta: Dict[str, Any]) -> Tuple[torch.Tensor, float, float]:
        """
        Executes SwiGLU forward:
          E(x) = down_proj( silu(gate_proj(x)) * up_proj(x) )
        Separate timing for dequantize/unpack vs pure GEMM.
        """
        precision = meta.get("precision", "bf16")
        t_dec_0 = time.perf_counter()

        if precision == "bf16":
            gate_w = gpu_data["gate_proj"]
            up_w = gpu_data["up_proj"]
            down_w = gpu_data["down_proj"]
        elif precision.startswith("int8"):
            group_size = meta.get("group_size", 128)
            gate_w = dequantize_int8_g128(gpu_data["gate_proj"], gpu_data["gate_scales"], group_size)
            up_w = dequantize_int8_g128(gpu_data["up_proj"], gpu_data["up_scales"], group_size)
            down_w = dequantize_int8_g128(gpu_data["down_proj"], gpu_data["down_scales"], group_size)
        elif precision.startswith("int4"):
            group_size = meta.get("group_size", 128)
            gate_w = dequantize_int4_g128(gpu_data["gate_proj"], gpu_data["gate_scales"], gpu_data["gate_shape"], group_size)
            up_w = dequantize_int4_g128(gpu_data["up_proj"], gpu_data["up_scales"], gpu_data["up_shape"], group_size)
            down_w = dequantize_int4_g128(gpu_data["down_proj"], gpu_data["down_scales"], gpu_data["down_shape"], group_size)
        else:
            raise ValueError(f"Unsupported precision: {precision}")

        if self.device == "cuda":
            torch.cuda.synchronize()
        t_dec_1 = time.perf_counter()
        decode_ms = (t_dec_1 - t_dec_0) * 1000.0

        # GEMM Timing
        t_gemm_0 = time.perf_counter()
        gate_out = torch.matmul(x, gate_w.T)
        silu_out = torch.nn.functional.silu(gate_out)
        up_out = torch.matmul(x, up_w.T)
        intermediate = silu_out * up_out
        out = torch.matmul(intermediate, down_w.T)

        if self.device == "cuda":
            torch.cuda.synchronize()
        t_gemm_1 = time.perf_counter()
        gemm_ms = (t_gemm_1 - t_gemm_0) * 1000.0

        return out, decode_ms, gemm_ms

    def get_stats(self) -> Dict[str, Any]:
        with self.lock:
            s = dict(self.stats)
            reqs = max(s["requests"], 1)
            s["vram_hit_rate"] = round(s["vram_hits"] / reqs, 4)
            s["ram_hit_rate"] = round(s["ram_hits"] / reqs, 4)
            s["ssd_miss_rate"] = round(s["ssd_reads"] / reqs, 4)
            s["target_hit_rate"] = round(s["target_precision_hits"] / reqs, 4)
            s["any_hit_rate"] = round(s["expert_any_precision_hits"] / reqs, 4)
            s["vram_used_mib"] = round(self.vram_used / (1024 * 1024), 2)
            s["ram_used_mib"] = round(self.ram_used / (1024 * 1024), 2)
            return s
