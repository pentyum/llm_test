#!/usr/bin/env python3
"""
P4.5 Microbenchmarks for I/O Tiers & GPU Operations
Measures:
  1. I/O Tiers:
     - Warm Linux Page Cache read
     - Attempted Cold NVMe read (via posix_fadvise DONTNEED)
     - Host-to-Device Copy (Pageable vs Pinned memory)
  2. GPU Compute & Memory:
     - Pure VRAM lookup
     - Dequantize / Unpack kernel time (CUDA events)
     - SwiGLU GEMM compute time (CUDA events)
     - Allocator peak & memory overhead
Outputs:
  - results/p45_io_microbench.csv
  - results/p45_gpu_microbench.csv
  - results/p45_latency_distributions.png
"""

import os
import sys
import time
import ctypes
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.weight_store import WeightStore
from src.quant_codec import dequantize_int8_g128, dequantize_int4_g128

RESULTS_DIR = "/home/yaoyao/llm_test/results"
IO_CSV = os.path.join(RESULTS_DIR, "p45_io_microbench.csv")
GPU_CSV = os.path.join(RESULTS_DIR, "p45_gpu_microbench.csv")
PLOT_PNG = os.path.join(RESULTS_DIR, "p45_latency_distributions.png")

os.makedirs(RESULTS_DIR, exist_ok=True)

POSIX_FADV_DONTNEED = 4

def drop_file_cache(filepath: str):
    """Attempt to evict file from Linux Page Cache using posix_fadvise."""
    try:
        fd = os.open(filepath, os.O_RDONLY)
        libc = ctypes.CDLL("libc.so.6")
        libc.posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED)
        os.close(fd)
    except Exception:
        pass

def benchmark_io_tiers(store: WeightStore, num_repeats: int = 15):
    print("=== Running I/O Tiers Microbenchmark ===")
    precisions = ["int4_g128", "int8_g128", "bf16"]
    records = []

    device = "cuda" if torch.cuda.is_available() else "cpu"

    for prec in precisions:
        fpath = store.get_expert_path(0, 0, prec)
        f_bytes = os.path.getsize(fpath)
        size_mib = f_bytes / (1024 * 1024)

        # 1. Warm Page Cache Read (read twice to ensure hot cache)
        store.load_expert(0, 0, prec)
        warm_times = []
        for _ in range(num_repeats):
            t0 = time.perf_counter_ns()
            data, _, _, _ = store.load_expert(0, 0, prec)
            t1 = time.perf_counter_ns()
            warm_times.append((t1 - t0) / 1e6)

        # 2. Attempted Cold Read (drop cache before each read)
        cold_times = []
        for _ in range(num_repeats):
            drop_file_cache(fpath)
            t0 = time.perf_counter_ns()
            data, _, _, _ = store.load_expert(0, 0, prec)
            t1 = time.perf_counter_ns()
            cold_times.append((t1 - t0) / 1e6)

        # 3. Host to Device Copy (Pageable)
        h2d_pageable_times = []
        for _ in range(num_repeats):
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter_ns()
            gpu_data = {k: v.to(device, non_blocking=False) if isinstance(v, torch.Tensor) else v for k, v in data.items()}
            if device == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter_ns()
            h2d_pageable_times.append((t1 - t0) / 1e6)

        # 4. Host to Device Copy (Pinned Memory)
        pinned_data = {}
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                pinned_data[k] = v.pin_memory()
            else:
                pinned_data[k] = v

        h2d_pinned_times = []
        for _ in range(num_repeats):
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter_ns()
            gpu_data = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in pinned_data.items()}
            if device == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter_ns()
            h2d_pinned_times.append((t1 - t0) / 1e6)

        for mode, t_list in [
            ("warm_page_cache", warm_times),
            ("attempted_cold_ssd", cold_times),
            ("h2d_copy_pageable", h2d_pageable_times),
            ("h2d_copy_pinned", h2d_pinned_times)
        ]:
            arr = np.array(t_list)
            mean_ms = float(np.mean(arr))
            bandwidth_gbps = (size_mib / 1024) / (mean_ms / 1000) if mean_ms > 0 else 0
            records.append({
                "precision": prec,
                "tier_mode": mode,
                "file_size_mib": round(size_mib, 3),
                "mean_ms": round(mean_ms, 4),
                "p50_ms": round(float(np.percentile(arr, 50)), 4),
                "p95_ms": round(float(np.percentile(arr, 95)), 4),
                "bandwidth_gbps": round(bandwidth_gbps, 2),
                "samples": num_repeats
            })

    df = pd.DataFrame(records)
    df.to_csv(IO_CSV, index=False)
    print(f"I/O microbenchmark saved to {IO_CSV}")
    print(df.to_markdown())
    return df

def benchmark_gpu_compute(store: WeightStore, num_repeats: int = 50):
    print("\n=== Running GPU Compute Microbenchmark ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("CUDA unavailable, skipping GPU microbenchmark")
        return pd.DataFrame()

    precisions = ["int4_g128", "int8_g128", "bf16"]
    records = []

    # Test activation: batch=16 tokens x 2048 hidden
    x = torch.randn(16, 2048, dtype=torch.bfloat16, device=device)

    for prec in precisions:
        data_cpu, meta, _, f_bytes = store.load_expert(0, 0, prec)
        gpu_data = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in data_cpu.items()}
        torch.cuda.synchronize()

        group_size = meta.get("group_size", 128)

        # 1. Dequantize Timing (CUDA Events)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        dec_times = []
        if prec == "bf16":
            # BF16 has 0 decode overhead
            dec_times = [0.0] * num_repeats
            gate_w = gpu_data["gate_proj"]
            up_w = gpu_data["up_proj"]
            down_w = gpu_data["down_proj"]
        elif prec == "int8_g128":
            for _ in range(num_repeats):
                start_event.record()
                gate_w = dequantize_int8_g128(gpu_data["gate_proj"], gpu_data["gate_scales"], group_size)
                up_w = dequantize_int8_g128(gpu_data["up_proj"], gpu_data["up_scales"], group_size)
                down_w = dequantize_int8_g128(gpu_data["down_proj"], gpu_data["down_scales"], group_size)
                end_event.record()
                torch.cuda.synchronize()
                dec_times.append(start_event.elapsed_time(end_event))
        elif prec == "int4_g128":
            for _ in range(num_repeats):
                start_event.record()
                gate_w = dequantize_int4_g128(gpu_data["gate_proj"], gpu_data["gate_scales"], gpu_data["gate_shape"], group_size)
                up_w = dequantize_int4_g128(gpu_data["up_proj"], gpu_data["up_scales"], gpu_data["up_shape"], group_size)
                down_w = dequantize_int4_g128(gpu_data["down_proj"], gpu_data["down_scales"], gpu_data["down_shape"], group_size)
                end_event.record()
                torch.cuda.synchronize()
                dec_times.append(start_event.elapsed_time(end_event))

        # 2. Pure GEMM Timing (CUDA Events)
        gemm_times = []
        for _ in range(num_repeats):
            start_event.record()
            g_out = torch.matmul(x, gate_w.T)
            u_out = torch.matmul(x, up_w.T)
            inter = torch.nn.functional.silu(g_out) * u_out
            out = torch.matmul(inter, down_w.T)
            end_event.record()
            torch.cuda.synchronize()
            gemm_times.append(start_event.elapsed_time(end_event))

        dec_arr = np.array(dec_times)
        gemm_arr = np.array(gemm_times)
        total_arr = dec_arr + gemm_arr

        records.append({
            "precision": prec,
            "mean_decode_ms": round(float(np.mean(dec_arr)), 4),
            "p50_decode_ms": round(float(np.percentile(dec_arr, 50)), 4),
            "p95_decode_ms": round(float(np.percentile(dec_arr, 95)), 4),
            "mean_gemm_ms": round(float(np.mean(gemm_arr)), 4),
            "p50_gemm_ms": round(float(np.percentile(gemm_arr, 50)), 4),
            "p95_gemm_ms": round(float(np.percentile(gemm_arr, 95)), 4),
            "mean_total_compute_ms": round(float(np.mean(total_arr)), 4),
            "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2),
            "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / (1024 * 1024), 2)
        })

    df = pd.DataFrame(records)
    df.to_csv(GPU_CSV, index=False)
    print(f"GPU microbenchmark saved to {GPU_CSV}")
    print(df.to_markdown())
    return df

def plot_microbench_charts(df_io: pd.DataFrame, df_gpu: pd.DataFrame):
    plt.figure(figsize=(15, 5))

    # Subplot 1: I/O Tiers Latency (Log scale)
    plt.subplot(1, 3, 1)
    modes = ["h2d_copy_pinned", "h2d_copy_pageable", "warm_page_cache", "attempted_cold_ssd"]
    for prec in ["int4_g128", "int8_g128", "bf16"]:
        sub = df_io[df_io["precision"] == prec].set_index("tier_mode")
        vals = [sub.loc[m, "mean_ms"] for m in modes]
        plt.plot(modes, vals, "o-", label=prec)
    plt.yscale("log")
    plt.ylabel("Mean Latency (ms, Log Scale)")
    plt.title("I/O Tier Latency Hierarchy")
    plt.xticks(rotation=25, ha="right")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    # Subplot 2: GPU Breakdown (Decode vs GEMM)
    plt.subplot(1, 3, 2)
    precs = df_gpu["precision"].tolist()
    dec_ms = df_gpu["mean_decode_ms"].tolist()
    gemm_ms = df_gpu["mean_gemm_ms"].tolist()
    x = np.arange(len(precs))
    plt.bar(x, gemm_ms, label="Pure GEMM", color="#2a9d8f")
    plt.bar(x, dec_ms, bottom=gemm_ms, label="Dequant / Unpack", color="#e76f51")
    plt.xticks(x, precs)
    plt.ylabel("Time (ms)")
    plt.title("GPU Execution Breakdown (Batch=16)")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    # Subplot 3: Effective Transfer Bandwidth
    plt.subplot(1, 3, 3)
    bw_pinned = df_io[df_io["tier_mode"] == "h2d_copy_pinned"].set_index("precision")["bandwidth_gbps"]
    bw_pageable = df_io[df_io["tier_mode"] == "h2d_copy_pageable"].set_index("precision")["bandwidth_gbps"]
    bw_cold = df_io[df_io["tier_mode"] == "attempted_cold_ssd"].set_index("precision")["bandwidth_gbps"]
    
    x = np.arange(len(precs))
    width = 0.25
    plt.bar(x - width, [bw_pinned.get(p, 0) for p in precs], width=width, label="Pinned H2D", color="#457b9d")
    plt.bar(x, [bw_pageable.get(p, 0) for p in precs], width=width, label="Pageable H2D", color="#a8dadc")
    plt.bar(x + width, [bw_cold.get(p, 0) for p in precs], width=width, label="Cold NVMe", color="#e63946")
    plt.xticks(x, precs)
    plt.ylabel("Bandwidth (GB/s)")
    plt.title("Effective Transfer Bandwidth")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(PLOT_PNG, dpi=200)
    print(f"Chart saved to {PLOT_PNG}")

def main():
    store = WeightStore()
    df_io = benchmark_io_tiers(store)
    df_gpu = benchmark_gpu_compute(store)
    plot_microbench_charts(df_io, df_gpu)
    print("=== Phase C Microbenchmarks Completed Successfully ===")

if __name__ == "__main__":
    main()
