#!/bin/bash
set -ex
source /home/yaoyao/miniconda3/etc/profile.d/conda.sh
conda activate moe-precision-lab

mkdir -p /home/yaoyao/llm_test/models/gguf

python - <<'PY'
import os
import sys
from huggingface_hub import hf_hub_download

repo_id = "allenai/OLMoE-1B-7B-0125-Instruct-GGUF"
filename = "OLMoE-1B-7B-0125-Instruct-Q4_K_M.gguf"
local_dir = "/home/yaoyao/llm_test/models/gguf"

print(f"Downloading {filename} from {repo_id}...")
path = hf_hub_download(
    repo_id=repo_id,
    filename=filename,
    local_dir=local_dir,
    local_dir_use_symlinks=False
)
size_gb = os.path.getsize(path) / (1024**3)
print(f"Downloaded successfully: {path} ({size_gb:.2f} GB)")
PY

ls -lh /home/yaoyao/llm_test/models/gguf/*.gguf
