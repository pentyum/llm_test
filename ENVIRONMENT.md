# 实验环境记录 (ENVIRONMENT.md)

## 1. 硬件配置

| 组件 | 详细参数 | 备注 |
|---|---|---|
| **主机/系统** | Ubuntu 26.04.1 LTS (Resolute Raccoon), Linux 7.0.0-42-generic x86_64 | YAOYAO-TIANXUAN |
| **GPU** | NVIDIA GeForce RTX 4060 Laptop GPU (PCI Bus: `0000:01:00.0`) | 8188 MiB VRAM (~8 GB GDDR6) |
| **GPU 驱动** | NVIDIA Driver 595.104.02 | 支持 CUDA 13.2 Driver API |
| **系统内存** | 46 GiB RAM (~48 GB Physical RAM) | 空闲/缓存 ~37 GiB 可用 |
| **Swap 空间** | 63 GiB Swap (nvme0n1p1 32G + nvme1n1p2 32G) | 充足交换分区 |
| **工作磁盘** | ZHITAI TiPlus7100 4TB NVMe SSD (`/dev/nvme0n1p3`) | 挂载于 `/home`，文件系统 `ext4` |
| **工作目录** | `/home/yaoyao/llm_test` | NVMe SSD 高速存储，剩余空间约 **2.1 TB** |
| **系统盘** | WD PC SN560 1TB NVMe SSD (`/dev/nvme1n1p3`) | 挂载于 `/`，文件系统 `ext4`，剩余 812 GB |
| **冷存储盘** | WDC WD40EMAZ-11LW3B0 4TB HDD (`/dev/sda1`) | 挂载于 `/mnt/data`，ext4，剩余 1.2 TB |

## 2. 软件与工具链

| 工具 | 路径 / 版本 | 说明 |
|---|---|---|
| **Conda** | `/home/yaoyao/miniconda3/bin/conda` (conda 24.11.3) | Conda 环境管理 |
| **Conda 环境** | `moe-precision-lab` (`/home/yaoyao/miniconda3/envs/moe-precision-lab`) | Python 3.11 独立环境 |
| **Python** | Python 3.11.18 | Conda 内部环境 |
| **CUDA Toolkit / nvcc** | `/usr/bin/nvcc` (release 12.4, V12.4.131) | 系统 CUDA 12.4 工具链 |
| **CMake** | `/usr/bin/cmake` (version 4.2.3) | 支持 C++20 / Ninja 构建 |
| **PyTorch** | PyTorch 2.6.0+cu124 (CUDA 12.4) | 官方 prebuilt wheel |
| **Hugging Face CLI** | `huggingface-cli` (huggingface_hub) | 模型下载与管理 |

## 3. 实验契约核对

- [x] 主实验设备为 RTX 4060 (8188 MiB VRAM)
- [x] 磁盘可用空间 (2.1 TB) 远高于 10 GiB 门禁标准
- [x] 独立 Conda 环境 `moe-precision-lab`
- [x] 工作目录位于高速 NVMe SSD
