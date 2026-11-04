#!/bin/bash
set -ex
source /home/yaoyao/miniconda3/etc/profile.d/conda.sh
conda activate moe-precision-lab

cd /home/yaoyao/llm_test

echo "=== 1. Running All P5a.1 Unit Tests ==="
PYTHONPATH=. pytest -v tests/p5a1/

echo "=== 2. Running T0-T6 Matrix and Validation ==="
python scripts/run_p5a1_matrix.py

echo "=== P5a.1 Validation Complete ==="
