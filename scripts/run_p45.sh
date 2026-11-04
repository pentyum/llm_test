#!/bin/bash
set -ex
source /home/yaoyao/miniconda3/etc/profile.d/conda.sh
conda activate moe-precision-lab

cd /home/yaoyao/llm_test

echo "=== 1. Running Unit Tests ==="
PYTHONPATH=. pytest -v tests/

echo "=== 2. Running Microbenchmarks (I/O & GPU) ==="
python scripts/microbench_io_gpu.py

echo "=== 3. Extracting Real Router Trace & Sensitivity ==="
python src/extract_real_router_trace.py

echo "=== 4. Running 5-Repeat Policy Benchmark ==="
python scripts/run_p45_experiments.py

echo "=== P4.5 Run Complete! ==="
