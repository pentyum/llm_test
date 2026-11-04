# P5b.1 可复现性与运行步骤指南 (P5B1_REPRO_STEPS.md)

本文档提供在全新环境或任意开发机上 100% 独立复现 P5b.1 全部实验结果的操作步骤。

---

## 1. 环境准备

```bash
cd /path/to/llm_test
source /home/yaoyao/miniconda3/etc/profile.d/conda.sh
conda activate moe-precision-lab

# 依赖版本锁定
# Python 3.11.18, PyTorch 2.6.0+cu124, Transformers 5.22.1
```

---

## 2. 离线专家多精度权重库校验

```bash
# 验证 4 个关键层 [0, 4, 8, 12] 的全部 768 个预存权重完整无损
python scripts/p5b/verify_store.py
```

---

## 3. 运行自动化单元测试套件

```bash
# 运行 P5b.1 全部单元测试（包含 attach/restore 安全语义、预取任务状态机、多层 BF16/Router 数值回归等）
PYTHONPATH=. pytest -v tests/p5b1/
```

---

## 4. 运行严格公平的预取消融实验

```bash
# 在固定 INT4 与固定 INT8 下严格对照 sync_no_prefetch vs static_layer_prefetch vs oracle_prefetch
python scripts/p5b1/run_ablation.py
```

---

## 5. 自动构建总结报告

```bash
# 聚合全部生成与消融数据，自动更新阶段门禁判定
python scripts/p5b1/build_report.py
```
