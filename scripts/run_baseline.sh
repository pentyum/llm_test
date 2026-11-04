#!/bin/bash
set -ex
source /home/yaoyao/miniconda3/etc/profile.d/conda.sh
conda activate moe-precision-lab

cd /home/yaoyao/llm_test
python scripts/run_p1_experiments.py
