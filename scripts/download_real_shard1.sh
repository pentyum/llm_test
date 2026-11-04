#!/bin/bash
set -ex
source /home/yaoyao/miniconda3/etc/profile.d/conda.sh
conda activate moe-precision-lab

mkdir -p /home/yaoyao/llm_test/models/hf_original

python - <<'PY'
import os
from huggingface_hub import hf_hub_download

repo_id = "allenai/OLMoE-1B-7B-0125-Instruct"
filename = "model-00001-of-00003.safetensors"
local_dir = "/home/yaoyao/llm_test/models/hf_original"

print(f"Downloading {filename}...")
p = hf_hub_download(
    repo_id=repo_id,
    filename=filename,
    local_dir=local_dir
)
size_gb = os.path.getsize(p) / (1024**3)
print(f"Downloaded successfully: {p} ({size_gb:.2f} GB)")
PY

ls -lh /home/yaoyao/llm_test/models/hf_original/
