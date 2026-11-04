#!/bin/bash
set -ex
source /home/yaoyao/miniconda3/etc/profile.d/conda.sh
conda activate moe-precision-lab

cd /home/yaoyao/llm_test

echo "=== Running P5a Single-Layer Dynamic MoE Integration ==="
python scripts/run_p5a.py

echo "=== P5a Run Complete! ==="
