#!/bin/bash
set -ex
source /home/yaoyao/miniconda3/etc/profile.d/conda.sh
conda activate moe-precision-lab

echo "=== Installing PyTorch ==="
pip install torch --index-url https://download.pytorch.org/whl/cu124

echo "=== Installing dependencies ==="
pip install huggingface_hub safetensors transformers accelerate numpy pandas scipy matplotlib pyyaml psutil pytest tqdm

echo "=== Verifying PyTorch and CUDA ==="
python - <<'PY'
import torch
print('torch:', torch.__version__)
print('torch CUDA:', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('VRAM GB:', round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2))
    # Quick matmul test
    a = torch.randn(1024, 1024, device='cuda')
    b = torch.randn(1024, 1024, device='cuda')
    c = torch.matmul(a, b)
    torch.cuda.synchronize()
    print('CUDA matmul test passed! Result norm:', c.norm().item())
PY
echo "=== Environment Setup Complete ==="
