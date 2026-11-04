#!/bin/bash
set -ex
source /home/yaoyao/miniconda3/etc/profile.d/conda.sh
conda activate moe-precision-lab

cd /home/yaoyao/llm_test

echo "=== 1. Running P5a.3 Unit Tests ==="
PYTHONPATH=. pytest -v tests/p5a3/

echo "=== 2. Auditing BF16 Weight and Router Parity (G1) ==="
python scripts/run_p5a3_parity.py --config configs/p5a3/default.yaml

echo "=== 3. Evaluating Teacher-Forced Logits (G2) ==="
python scripts/run_p5a3_teacher_forced.py

echo "=== 4. Aggregating Results and Generating P5A3_REPORT.md (G9) ==="
python scripts/aggregate_p5a3.py --input results/p5a3 --output docs/p5a3/P5A3_REPORT.md

echo "=== P5a.3 Completed Successfully! ==="
