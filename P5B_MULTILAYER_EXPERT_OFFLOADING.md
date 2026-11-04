# P5b 实验计划：多层 MoE 动态混合精度与真实 SSD 专家卸载

> **执行对象**：Coding Agent（可直接执行）  
> **项目**：[`pentyum/llm_test`](https://github.com/pentyum/llm_test)  
> **基线分支**：`main`（执行时记录实际 commit SHA；不要假定下面引用的旧 SHA 仍为 HEAD）  
> **建议新分支**：`experiment/p5b-multilayer-offloading`  
> **主要设备**：Ubuntu + NVIDIA RTX 4060 Laptop 8 GB VRAM + 48 GB RAM + NVMe SSD  
> **Conda**：优先复用项目中已经验证的环境；如需隔离，创建 `moe-p5b` 新环境  
> **目标模型**：`allenai/OLMoE-1B-7B-0125-Instruct`  
> **版本**：P5b v1.0 · 2026-11-04

---

## 0. 给 Agent 的顶层指令（必须遵守）

1. **先修正 P5a.3 证据链，再扩展多层，最后才允许宣称 SSD 卸载有效。** 不可跳过门禁。
2. 研究目标是 **SSD 上离线预存多个精度版本 → 在线按专家选择精度 → SSD/RAM/VRAM 分层缓存和计算**。**禁止在线重新量化**。在线允许解包/反量化，但必须分别计时并明确不等于原生 INT4 GEMM。
3. **优先 RTX 4060**；不因一次 OOM 自动改用 RTX 5080。先减小 cache、batch、context、选定层数或调整不参与测试的权重驻留方式。任何回退都必须记录。
4. 保留 P1–P5a.3 所有历史结果；新增成果仅写入 `results/p5b/`、`docs/p5b/` 等新目录。禁止悄悄覆盖历史 CSV/JSON/图表。
5. 绝不把 **逻辑 SSD 请求字节**、**Linux Page Cache 命中**、**真实块设备读取量** 混为一谈；不把 **单专家微基准**、**局部单层推理** 和 **端到端 Token 吞吐** 混为一谈。
6. 所有实验结论由**本次原始日志与配置自动计算生成**。若代码/报告中的统计不一致，标记 `FAIL`，不允许手工写成 `PASS`。
7. 对于无法在本设备完成的环节，明确列出阻碍、复现步骤和已完成范围；**允许诚实的部分成功**，不得虚构吞吐提升、精度变化或 SSD 实读。
8. 原则上在新分支上提交代码、配置、测试和小体积实验摘要。不要将模型权重、完整 HF cache、数 GB expert blob、访问令牌提交到 Git。不要强推或清理用户已有文件。

### 本轮验收的核心问题

> 在严格的 8 GB VRAM 预算下，使用 2–4 层（再扩展至更多层）真实 OLMoE 专家，以 **SSD 多版本预存储 + 按需精度选择 + 分层缓存 + 可选预取** 进行**真实增量 Decode**，是否能在**可解释的质量代价**下，改善固定精度策略的**延迟—内存—SSD I/O**权衡？

**重要**：此阶段并不要求动态 S4 必须比所有固定精度都快。要求的是可验证的实际数据与公平 Pareto 比较；达不到收益也应明确报告。

---

## 1. 项目现状和已知风险（依据仓库 P5a.3 报告；不是本轮新测量）

已公开的 P5a.3 报告：[`docs/p5a3/P5A3_REPORT.md`](https://github.com/pentyum/llm_test/blob/main/docs/p5a3/P5A3_REPORT.md)。报告注明基线 commit `bc1a92ae1cd8cf3ce92c6cfd17bfb4c808167955`，执行时必须以实际 `git rev-parse HEAD` 为准。

| 已记录结果 | 报告值 | P5b 的处理方式 |
|---|---:|---|
| Layer 0 专家 BF16 张量映射 | 192 / 192 逐位一致 | 保留为回归测试；新增替换层也要检查 |
| Teacher-forced BF16 Top-1 agreement | 98.63% | 不是位级模型等价；对比需控制设备与内核 |
| 32 prompts × 3 repeats × 5 modes | 480 次生成 | 验证实际输出文件计数、提示词类别、运行标识 |
| S4 平均生成速度 | 12.20 tokens/s | 不得使用报告摘要中的 `12.8+` 代替 |
| S4 平均 token 匹配率 | 88.67% | 不得使用报告摘要中的 `93%+` 代替 |
| S4 切换事件 | 87 次 | 细分精度方向、同专家/不同专家、是否真实执行 |
| S4 INT8 占比 | 0% | 必须报告完整 INT4/INT8/BF16 比例 |

**P5a.3 已知问题（本轮必须先复核）**：

- 报告的描述性摘要与自身数据表不一致（`12.8+ t/s`、`93%+` vs `12.20 t/s`、`88.67%`），说明报告自动校验不足。
- 部分所谓 token 级延迟数据仍可能是“每次生成的摘要”，不足以重算全体 token 的 p95/p99；本轮必须保存**每个 decode token 一行**。
- BF16 权重映射正确不意味着 CPU/GPU 不同计算路径具有完全相同的 logits；不能用自回归字符串匹配率单独定义模型精度。
- 当前主要完成单层专家替换。**多层同时运行、有限内存下的专家 SSD 冷读及异步重叠**尚未得到 P5b 证据。

---

## 2. 实验范围、明确不做的事

### 本轮必做

- `G0`：代码与环境冻结、P5a.3 报告自动一致性检查；纠正历史报告时另存修订说明，不覆盖原始文件。
- `G1`：`Layer 0` → `Layer 0+4` → `Layer 0+4+8+12` 的真实多层专家替换与正确性测试。
- `G2`：建立**可独立随机读取**的多精度 SSD 专家仓库（INT4、INT8、BF16），索引以 `(layer, expert, precision, format_version)` 为键。
- `G3`：可严格执行字节预算的 RAM/VRAM 缓存与事件日志；确认 SSD-only、RAM-hit、VRAM-hit 的数据路径。
- `G4`：真实模型的同步 SSD 专家卸载和端到端 Decode 对照。
- `G5`：独立设计并测量异步预取（允许预取失败）；区分 oracle 上界与可部署预测策略。
- `G6`：公平对比固定 INT4、固定 INT8、动态 S4、多精度消融，测量质量与系统性能。
- `G7`：自动报告和完整测试、门禁判定。

### 本轮不强制

- 编写原生 Tensor Core INT4 GEMM 或 Triton 融合算子；可以实验，但不得阻塞数据路径正确性。
- 把全部 16 层一次性动态替换；若 4 层能稳定通过后，可作为可选扩展。
- 多机分布式、训练/微调、量化感知训练、动态 KV Cache 量化、CPU 专家执行引擎。
- 在有意保持 SSD 冷读时追求极限 tokens/s；**先证明路径正确，再衡量性能**。

---

## 3. G0 — 冻结、审计与可信度门禁（必须首先通过）

### G0.1 初始化

```bash
set -euo pipefail
cd /path/to/llm_test
git fetch origin
git switch main
git pull --ff-only origin main
git switch -c experiment/p5b-multilayer-offloading
mkdir -p configs/p5b docs/p5b results/p5b logs/p5b

git rev-parse HEAD | tee results/p5b/base_commit.txt
nvidia-smi | tee results/p5b/nvidia_smi.txt
uname -a | tee results/p5b/uname.txt
lsblk -o NAME,SIZE,TYPE,TRAN,MOUNTPOINTS | tee results/p5b/disks.txt
free -h | tee results/p5b/memory.txt
```

> 如已存在同名分支，先检查其提交与未提交改动，不要重置或覆盖。所有示例路径应根据当前仓库实际结构调整，不要假设某个函数或文件仍与上一轮相同。

### G0.2 Conda 检查

优先使用已运行 P5a.3 的 Conda 环境并记录 `conda list --explicit`。仅当缺依赖或包冲突时，执行以下命令：

```bash
conda create -n moe-p5b python=3.11 -y
conda activate moe-p5b
python -m pip install -U pip pytest pandas numpy matplotlib psutil safetensors pyyaml
# PyTorch、Transformers、CUDA/Triton 版本必须参考本机旧环境与项目锁定文件；
# 不要无理由升级至尚未验证的新版本。
```

必须记录 `torch.__version__`、`transformers.__version__`、CUDA runtime、GPU 名称、CPU、RAM、SSD 型号、文件系统、可用磁盘空间、`nvidia-smi` 实际可用显存、模型文件哈希/修订版本。**不要把上一轮机器读数当成今天实测**。

### G0.3 报告一致性修复

实现 `scripts/p5b/validate_previous_report.py`（若仓库已有通用报表验证器，复用它），要求：

1. 读取 `results/p5a3/quality/summary.csv`、`results/p5a3/quality/teacher_forced_logits.csv` 等真实存在的文件。
2. 解析并比对 `docs/p5a3/P5A3_REPORT.md` 的**数字型声明与表格**；不能自动可靠解析的声明列入人工复核清单。
3. 对 `12.8+` vs `12.20`、`93%+` vs `88.67%` 这类冲突必须输出 `FAIL`，另写 `docs/p5b/P5A3_CORRECTIONS.md`，保留来源、差异、原始结果位置。不要静默覆盖历史报告。
4. 用 `run_id`、`git_sha`、`model_revision`、`config_hash`、`source_csv` 给每张新报表建立可追溯链。
5. 在自动生成的 P5b 报告中，摘要必须从 CSV/JSON 加载，不允许写死“提升 X%”、“全部通过”。

**G0 验收**：

- `pytest` 原有测试完整通过（任何失败需归档）；
- 报告校验器能够故意抓出上面两组不一致数字；
- 新运行目录具备唯一 `run_id`、Git SHA、环境清单、config 和日志；
- 不需要删除或修改原来的实验输出。

**G0 不通过则停止后续大规模实验。** 可提交修复和问题记录。

---

## 4. G1 — 多层真实 MoE 替换（先不引入 SSD）

### G1.1 分阶段选层

| 阶段 | 替换层 | 目的 |
|---|---|---|
| L1 | `[0]` | 重跑单层 P5a.3 回归，确保新框架没改变语义 |
| L2 | `[0, 4]` | 验证多个层同时替换、键空间不冲突 |
| L4 | `[0, 4, 8, 12]` | 覆盖早/中/后层，测试累积误差和显存 |
| L16（可选） | 所有 MoE 层 | 仅在四层全通过且资源允许时做 |

先读取模型 `config`，确认架构、MoE 层数量、各专家矩阵名称、Router 计算、Top-k 与返回张量形状；如果版本不同，动态发现而不是硬编码。**一个专家实例的唯一键是 `(layer_id, expert_id)`**，不允许再仅用 `expert_id` 索引缓存和敏感度。

### G1.2 先做 BF16 数值回归

- 在**同一个设备、同一 dtype、同一输入 hidden states** 下比较原层与 BF16 替换层：Router logits、Top-k IDs、归一化 Router 权重、每专家输出、MoE 聚合输出、后续模型 logits。
- 使用 batch=1/2，seq_len=1/4/16 等组合，并设置确定的随机种子；禁用 dropout，确保模型 eval。
- 对 BF16 输出记录 `max_abs`、`rel_l2`、`cosine`、`top1_agreement`；门槛由单层实测数值与执行路径决定，**不可用“百分百 token 一致”代替数值检查**。
- 对偏差设置明确的 blocker：权重映射错误、Router Top-k 不一致、shape 或残差连接错误、NaN/Inf、增量 KV Cache 失配，任一存在必须修复。
- 对仅因浮点累加顺序造成的稳定差异，提供同设备/不同设备对照及证据，允许经记录后继续。

### G1.3 精度与层数消融

至少执行：

- BF16 passthrough：L1、L2、L4；
- Fixed INT4：L1、L2、L4；
- Fixed INT8：L1、L2、L4；
- S4：先 L2，后 L4，记录动态选择是否实际发生。

继续使用增量 KV Cache，**不得每生成一个 token 都把完整历史序列重新喂入模型**。对 `KV Cache` 的位置、长度以及下一 token logits 做回归。

**G1 验收**：L2/L4 不出现专家跨层串用、权重精度错配、设备/shape 错误；在同输入下 BF16 数值回归满足记录的可解释容差；所有模式能完成至少 4 条 smoke prompts 的有限步生成，无 CUDA OOM/NaN。

---

## 5. G2 — SSD 多精度专家仓库（离线生成，一次保存）

### G2.1 设计可复现的存储格式

推荐逻辑目录（实际允许使用单个大文件 + offset 索引以减少海量小文件）：

```text
artifacts/p5b/expert_store/
  manifest.json
  layer_00/
    expert_000/
      bf16.safetensors
      int8_g128.safetensors
      int4_g128.safetensors
  layer_04/
  layer_08/
  layer_12/
```

**必须保留：**

- `layer_id`、`expert_id`、矩阵 `gate/up/down`、原始 shape、dtype；
- 量化格式、group_size、scale/zero-point 规则、排列布局、pack 版本、endianness、必要的 padding；
- 路径或大文件 offset、真实文件长度、SHA256 或内容校验；
- 来源模型 revision、原始权重哈希、离线量化脚本版本、量化校准信息；
- 与推理算子期望的格式兼容性标记。

不同精度的内容必须**在推理开始前**全部生成；在线路径不允许出现 `quantize()` 或写入新精度版本的行为。对于 BF16 可以共享原文件映射，但必须验证读出的确实是目标专家。

### G2.2 兼容性和原子性

- 离线生成使用 `tmp → fsync → atomic rename` 或等价安全写入；manifest 最后提交。
- 读取方只处理完整、校验通过的版本，不把损坏文件当正常 cache miss。
- 使用明确文件对齐策略；大文件方案记录每个 tensor 偏移与 length，避免单专家读取意外扫描全文件。
- 对模型变更或 store 版本变更设置 cache invalidation，绝不能复用旧版本的相同键。
- 使用离线随机抽检（至少多个层的若干专家）：INT4/INT8 解包后的张量形状与输出误差和此前测试在合理区间一致。

### G2.3 空间规划与缓存防误判

不要求把全部 BF16/INT8/INT4 专家同时载入 RAM。磁盘空间按实际导出的专家数计算，不把“单专家大小 × 1024”当成已测模型占用。使用文件索引记录总字节和实际占用 `du`。

**G2 验收**：在进程完全退出并重新启动之后，对随机 `(layer, expert, precision)` 可以从 SSD 重新加载，校验成功，且与离线原始版本一致；在线日志中量化调用次数为 0。

---

## 6. G3 — 严格预算的 VRAM/RAM 缓存与完整事件记录

### G3.1 数据路径必须明确

```text
PrecisionPolicy(layer, expert, router_score, sensitivity, cache_state)
                |
                v
       WeightKey(layer, expert, precision, format_version)
                |
       +--------+---------+
       |                  |
   VRAM hit              miss
       |                  |
    compute           RAM hit? ---- yes --> H2D async + compute
                          |
                          no
                          v
                      SSD read --> RAM staging --> H2D --> compute
```

其中 `VRAM hit` 表示**目标版本已在显存**，`RAM hit` 表示目标版本在系统内存，`other_precision_hit` 表示同专家其他精度版本被命中，三者不能混为“目标精度显存命中”。

### G3.2 缓存对象与容量

关键字段：

```python
@dataclass(frozen=True)
class WeightKey:
    layer_id: int
    expert_id: int
    precision: str
    format_version: str
    store_revision: str
```

必须维护：

- `vram_used_bytes <= vram_budget_bytes` 与 `ram_used_bytes <= ram_budget_bytes` 的硬约束；
- 每个对象的字节数包含**packed weights、scales、zero-points、辅助 metadata 的实际驻留数据**；临时 staging、反量化 BF16 矩阵、激活、KV Cache 另设峰值统计，不得伪装成专家缓存已覆盖的内存；
- 同一 key 的 `in_flight` 并发请求合并，异常可传递、可超时，失败不标为缓存命中；
- 缓存驱逐与 CUDA stream/event 依赖安全；执行中的权重不得释放或覆盖；
- 零预算、对象大于预算、并发竞争、OOM 恢复、`clear_cache` 都有单元测试。

### G3.3 可观测性——每次专家请求一条事件

建议最少字段：

```json
{
  "run_id": "...", "prompt_id": "...", "decode_step": 7,
  "layer_id": 4, "expert_id": 23,
  "router_score": 0.14, "precision_requested": "int8_g128",
  "precision_executed": "int8_g128", "precision_fallback_reason": null,
  "cache_source": "ssd|ram|vram", "other_precision_hit": false,
  "prefetch_requested": false, "prefetch_ready": false,
  "logical_read_bytes": 0, "process_read_bytes_delta": 0,
  "read_wall_ms": 0.0, "h2d_ms": 0.0, "decode_ms": 0.0,
  "expert_compute_ms": 0.0, "total_expert_wait_ms": 0.0,
  "vram_cache_used_bytes": 0, "ram_cache_used_bytes": 0,
  "event_status": "ok"
}
```

`process_read_bytes_delta` 应标注为**进程级近似归因**，不能保证多线程多进程背景 I/O 下等于某单个专家真实物理读取；使用全 run 的前后计数与 `iostat`/设备层计数交叉核验。原始事件按流式 JSONL 写入，不在内存无限积累。

### G3.4 RAM 与 SSD 的重要区别

Linux Page Cache 命中不等于真正 SSD 读取。需分别记录：

- 逻辑 payload 字节：存储层实际向读接口请求的大小；
- 文件系统缓冲命中：可根据受控实验推断/监测，不强行假装精确分摊到每个请求；
- 进程 `read_bytes` 与设备级 block reads：用于判断是否真的发生磁盘读取；
- H2D 的 CUDA Event 延迟（注意 stream 同步和 pinned memory 条件）；
- 实际专家执行耗时和端到端 TPOT。

**严禁默认执行 `echo 3 > /proc/sys/vm/drop_caches`**：可能影响整台电脑和其他工作。冷读可用独立数据文件、受控 `posix_fadvise(..., DONTNEED)`（不保证完全冷）、RAM 压力安全方法，或在明确记录的条件下做 `O_DIRECT` 对照（对齐与格式限制必须处理），同时验证真实 block I/O。**如果不能严格冷读，则如实标为 buffered read，不得冠名 cold SSD。**

**G3 验收**：100% 测试记录不超预算，VRAM/RAM/SSD 路径可区分；允许精度回退时记录原因；正常运行没有隐式在线量化、无无限等待/异常吞噬。

---

## 7. G4 — 真实同步 SSD 卸载（最重要的工程门禁）

### G4.1 确保真的卸载，而不是内存里复制一份

当前 OLMoE 全模型能够在 48 GB RAM 中驻留，因此**最容易出现的假阳性**是：测试说从 SSD 读取专家，但原始专家仍被 HF 模型对象引用，或数据全部落入 Linux Page Cache。

必须设计真实的权重所有权策略：

1. 保留模型执行需要的**非目标层权重、attention、Router 和其他参数**；
2. 被替换层的专家 BF16 原始参数不应继续由前台推理模型长久持有（若由于框架设计无法真正释放，必须记录 RSS/anon/pinned allocation 并在报告中标注“存在内存双份，不能代表完整卸载”）；
3. 目标层专家仅通过 `ExpertStore` 按精度请求；权重缓存被清空且进程重启后仍可完成推理；
4. 不要将“缓存用的 INT4 被释放”视作“原始 BF16 模型参数也释放”；这两个驻留生命周期需要独立核对。

可采用构造自定义 MoE Layer 后替换原模块，删除对旧专家的强引用，必要时 `gc.collect()`；谨慎处理 PyTorch caching allocator：`torch.cuda.memory_reserved`、`allocated`、进程 RSS、系统可用内存都要记录。不得用 `torch.cuda.empty_cache()` 当成释放仍被引用参数的证据。

### G4.2 一次只改变一个因素

首先执行 `L2 + fixed INT4 + sync SSD` 的 smoke test，再测 L4。对照组在完全相同 layer、prompt、KV Cache、生成长度、缓存预算下运行：

| 组别 | 源头 | 精度 | 目的 |
|---|---|---|---|
| B0 | 全 RAM 热驻留 | 固定 INT4 | 不需要 SSD 的性能上界（在本机可执行时） |
| B1 | SSD-only，0 RAM 缓存 + 有限 VRAM 缓存 | 固定 INT4 | 确认真实 SSD 读取开销 |
| B2 | SSD + RAM + VRAM 分层 | 固定 INT4 | 测量分层缓存价值 |
| B3 | SSD + RAM + VRAM 分层 | 固定 INT8 | 更高精度固定基线 |
| B4 | SSD + RAM + VRAM 分层 | S4 动态 | 测量联合调度收益/代价 |

BF16 full baseline 与 BF16 passthrough 可作为质量锚点，但若硬件无法公平驻留，不强行纳入同样的吞吐比较表。**不要把 B0 的全量内存使用和 B2/B4 的有限内存吞吐简单视为相同资源条件。**

### G4.3 首个真实 SSD 读取验收案例

使用可复现的“单 expert 文件真实读取” + “真实 decode 中至少一个 cache miss”的成对测试：

- 清空 **应用级** RAM/VRAM expert 缓存，重启测试进程；
- 记录原始专家参数是否仍驻留（CPU RSS anonymous / GPU allocated）；
- 选取将被请求的一个 layer/expert/precision；
- 记录确实从磁盘索引打开/读取该版本文件、校验 header/checksum、构造正确 GPU tensor；
- 用该专家完成一次真实 MoE 前向，输出与相同预量化权重的热加载路径一致；
- 若操作系统 Page Cache 掩盖设备 I/O，需要单独运行可控的真实物理 SSD 读对照；若仍无法证明物理 I/O，写 `SSD-backed logical load confirmed; physical cold read unverified`，不可写 `cold SSD PASS`。

**G4 验收**：L2、L4 至少能够完成固定 INT4 的真实增量 Decode；专家的 SSD→RAM→VRAM 路径有调用证据；缓存预算严格生效；在“真正由 SSD 提供专家数据”的条件下至少有可核验的块设备读取样本。若没做到，不能标为“SSD 卸载已验证”。

---

## 8. G5 — 异步预取：先做可行上界，再做可部署策略

### G5.1 不允许未来信息泄露

需要区分：

- `sync_no_prefetch`：标准基线；
- `oracle_prefetch`：利用**事先采集的同一条轨迹**提供未来专家，仅作为**理论上界**，不得称为可部署方案；
- `static_layer_prefetch`：基于历史热门专家、上一轮统计、层顺序，不读取未来真实 Router 结果；
- `predictive_prefetch`（可选）：根据允许使用的当前/过去信息预测后续专家。

如果某层 Router 尚未计算出来，不能把它在未来才产生的精确 Top-k 当作当前可见信息。任何先看真实后续轨迹的做法都必须明确写成 oracle。

### G5.2 重叠流水线

期望分工：

```text
Current layer GPU expert compute
    overlaps with RAM prefetch for next predicted layer
    overlaps with SSD read for future predicted expert(s)
```

- 预取有独立的、有限容量的任务队列与 in-flight 去重；
- 使用 async I/O / 线程池，H2D 使用单独 CUDA stream；若内存不是 pinned，记录其影响；
- GPU 使用前必须正确等待预取完成的 CUDA Event；
- 预取失败/预测失败/超预算都有可记录回退，并允许同步加载；
- 统计预取总字节、实际使用字节、浪费字节、命中率、发起到就绪时间、被隐藏的等待时间；
- 预取不能无界侵占 RAM/VRAM，必须和应用级缓存使用同一容量/优先级约束。

### G5.3 公平消融

同一层数（优先 L4）、相同 prompt、缓存预算、精度策略下比较：

- `sync_no_prefetch`
- `static_layer_prefetch`
- `oracle_prefetch`（上界）

记录 TTFT、token 级 TPOT p50/p95/p99、端到端 t/s、实际 SSD 读量、prefetch waste、缓存 miss 率、GPU 高水位内存。若预取明显变慢，也要报告和解释。

**G5 验收**：功能等价、无数据竞争/死锁；能证明预取字节实际计入 I/O；有不使用未来 Router 信息的真实策略结果。性能提升不是通过测试的先决条件。

---

## 9. G6 — S4 与固定精度的公平对照

### G6.1 必须明确比较对象

最低配置：

1. 原模型参考（BF16 或其可执行基准）：质量锚点。
2. 多层 `bf16_passthrough`：验证替换开销及数值偏差。
3. 多层 `fixed_int4`：低容量基线。
4. 多层 `fixed_int8`：高精度量化基线。
5. 多层 `s4_measured_cost`：动态策略。
6. `s4_no_cache_cost`：消融，不看缓存位置。
7. `s4_no_sensitivity`：消融，不看专家敏感度。

本轮可以额外加入 `int4_or_int8_only` 与 `int4_or_bf16_only`，确认此前 INT8 选择比例为零的原因；在质量约束可行时，记录各种精度实际执行比例和切换方向。

### G6.2 成本模型建议

不只看文件大小：

\[
C_i(b) = T_{\mathrm{unhidden\_load},i}(b) + T_{\mathrm{decode},i}(b) + T_{\mathrm{compute},i}(b) + \lambda \cdot \widehat{E}_i(b)
\]

其中 `T_unhidden_load` 是无法通过预取隐藏的等待时间，不应简单等于设备所有 I/O 时间相加。`E_hat` 只是局部敏感度代理，**不是最终模型质量上界**。必须允许固定质量预算、在线选择的计划成本和事后测得成本分别输出。

建议先用独立 microbenchmark 实测 `storage_tier × precision × model_layer` 成本表，保存均值、p50、p95、样本量、运行顺序和 cache 状态，再让 S4 查表/插值；若实测不稳定，保留区间或稳健代价，不用未经验证的单点估计。

### G6.3 精度切换日志

在相同 `(layer_id, expert_id)` 上，统计：

- `int4 → int8`、`int8 → int4`、`int4 → bf16`、`bf16 → int4`、其他方向；
- `requested_precision != executed_precision` 的回退数量和原因；
- 每个专家跨 decode step 的真实切换，不要把两个不同专家选了不同精度计为“同专家切换”；
- 每次切换是否因 miss 触发 SSD I/O、额外 H2D，及相应的 token 尾延迟；
- **至少展示一个同一专家真实自适应切换的事件片段（非强制测试人为切换）**。如果真实负载下未出现，诚实注明未观测到，不要硬编码报告。

### G6.4 质量与公平测量

- 使用仓库现有的 32 prompts（4 类 × 各 8 条），在合理硬件允许下执行**3 repeats、固定 seed**；每种模式同样的 prompt、max_new_tokens、context、温度/贪婪采样和 KV Cache 设置。
- 主质量指标：固定 teacher-forced prefix 下 mean/max KL、Top-1 agreement、logits rel L2；辅助指标：自由生成逐 token 匹配、输出文本、任务评分（若有明确标准答案）。
- 对 BF16 差异保留同设备控制试验；不将 Token 匹配率 100% 作为唯一正确性要求。
- 如测试总时间太长，先用预先声明的 4 条 smoke、8 条 pilot、32 条 full 三档；报告实际样本数，不可拿 pilot 数字写 full 结论。
- 每个对照模式重新建立独立缓存状态，严格区分 cold vs warm；**不要让前一个模式的缓存预热下一个模式**。
- 随机化模式运行顺序或使用轮换顺序，记录 GPU 时钟、温度与系统负载变化。

### G6.5 建议实验矩阵（避免组合爆炸）

| 批次 | 层数 | 策略 | RAM 专家缓存 | VRAM 专家缓存 | 预取 | 运行规模 |
|---|---|---|---|---|---|---|
| M0 smoke | L2 | fixed INT4 | 0 / 256 MiB | 128 MiB | off | 4 prompts |
| M1 sync | L4 | fixed INT4 / fixed INT8 / S4 | 0 / 256 / 1024 MiB | 128 / 256 MiB | off | 8 prompts |
| M2 prefetch | L4 | fixed INT4 / S4 | 256 MiB | 256 MiB | static / oracle | 8 prompts |
| M3 full | L4 | fixed INT4 / fixed INT8 / S4 + reference | 256 / 1024 MiB（择测量最有信息量者） | 256 MiB | off / static（择测量最有信息量者） | 32 prompts × 3 repeats |
| M4 optional | L16 | 最优的可行策略 | 受内存峰值约束 | 受内存峰值约束 | 可选 | 先 4 prompts |

**缓存预算是候选参数，不是强行要求**。首先测量 backbone、KV Cache 和临时缓冲占用，实际 VRAM 缓存预算不能挤占运行必需空间。若 L4 BF16 baseline 不可执行，必须说明资源不等价，不能造出对照数据。

---

## 10. G7 — 性能测量规范与报告生成

### G7.1 必须按 token 输出原始延迟

`results/p5b/<run_id>/decode_tokens.jsonl`：每个真实解码 token 一行，至少：

```json
{
  "run_id": "...", "mode": "s4_measured_cost",
  "prompt_id": "architecture_01", "repeat": 0, "decode_step": 12,
  "token_id": 123, "tpot_wall_ms": 81.3,
  "expert_wait_total_ms": 8.2,
  "gpu_compute_ms": 57.4,
  "ssd_logical_read_bytes_delta": 3145728,
  "vram_peak_allocated_bytes": 0,
  "vram_peak_reserved_bytes": 0,
  "num_precision_switches": 1,
  "status": "ok"
}
```

TTFT 与 TPOT 严格分开；第一 token 不应错误计入 decode TPOT；按 CUDA 同步边界或精确 CUDA Event 测量。不得仅保存**每次生成的平均 TPOT**却声称能重算所有 token 的 p95/p99。所有聚合统计从 `decode_tokens.jsonl` 计算。

### G7.2 SSD / RAM / VRAM 指标

最低指标：

| 类型 | 指标 |
|---|---|
| 延迟 | TTFT、decode TPOT mean/p50/p95/p99、总墙钟、tokens/s |
| SSD | 逻辑读取字节、进程 `read_bytes`、块设备读量（可得时）、读取次数、prefetch waste |
| 缓存 | 目标精度 VRAM/RAM 命中率、其他精度命中率、miss、eviction、峰值预算利用率 |
| 显存 | PyTorch `allocated`/`reserved`、整卡进程显存、峰值、KV Cache 所占 |
| 内存 | RSS、anonymous RSS、文件页映射、系统可用 RAM、page faults（可得时） |
| 计算 | INT4/INT8 解包或反量化、专家 GEMM、GPU stream 同步等待 |
| 质量 | Teacher-forced KL/Top-1、自由生成匹配率、精度选择与切换方向 |
| 实验环境 | SHA、config hash、model revision、seed、依赖版本、温度/时钟（可得时） |

### G7.3 异常与尾延迟归因

对 p95/p99 token，列出影响最大的请求（层、专家、精度、tier、是否预取、SSD miss 字节、h2d wait、计算耗时）。如果需要内核 profiling，优先在少量代表性 token 上使用 `torch.profiler`/Nsight，不要对全量评测开启高开销 trace。

### G7.4 统计与结论

- 所有定量图都同时保留原始 CSV/JSONL；用 **run ID** 链接。
- 对每个 `prompt_id` 做成对比较并报告差值分布；3 repeats 可给均值/标准差，但不能将三个重复视为大量独立 prompt 来夸大置信度。
- 用 latency vs quality 与 SSD traffic vs quality 的 Pareto 图，不强制指定赢家。
- 在最后报告中至少区分 **功能正确性**、**真实 SSD 数据路径**、**性能收益**、**质量改善** 四个独立 verdict。
- 如果某阶段未通过、某个硬件指标没测到或一个测量只是估计，显示 `FAIL` / `NOT VERIFIED` / `ESTIMATED`，不得移出表格。

---

## 11. 建议文件结构与产物契约

在不破坏现有结构的情况下，优先复用 `src/tiered_cache.py`、`src/weight_store.py`、`src/p5a/` 等已有代码。**以下为建议逻辑模块名；实际位置需 Agent 在首次 repo audit 后确认。**

```text
configs/p5b/
  smoke.yaml
  sync_offload.yaml
  prefetch.yaml
  full_eval.yaml

src/p5b/
  expert_store.py           # SSD 版本索引和随机读取
  offloaded_moe.py          # 多层专家替换与数据路径
  cache_adapter.py          # 可复用现有 tiered_cache
  prefetch_manager.py       # 同步/静态/oracle 预取
  quality_metrics.py        # teacher-forced 与自由生成
  event_logger.py           # expert & per-token 事件

scripts/p5b/
  audit_repo.py
  validate_previous_report.py
  export_expert_store.py
  verify_store.py
  run_multilayer.py
  benchmark_io.py
  run_offload.py
  run_prefetch_ablation.py
  evaluate_full.py
  generate_report.py
  validate_results.py

tests/p5b/
  test_manifest_and_integrity.py
  test_cache_capacity.py
  test_cache_concurrency.py
  test_multilayer_parity.py
  test_offline_quant_only.py
  test_ssd_path.py
  test_prefetch_semantics.py
  test_event_schema.py
  test_report_consistency.py

results/p5b/
  base_commit.txt
  ...environment files...
  <run_id>/
    config.resolved.yaml
    environment.json
    expert_events.jsonl
    decode_tokens.jsonl
    summary.csv
    quality.csv
    io_metrics.json
    stage_gates.json
  overview.csv
  plots/

docs/p5b/
  P5A3_CORRECTIONS.md
  ARCHITECTURE.md
  P5B_REPORT.md
  KNOWN_LIMITATIONS.md
```

**注意**：上述目录不意味着必须新建冗余代码。若当前仓库已实现相同功能，重构/包装优于复制多份逻辑。

---

## 12. 自动化测试：不可跳过的测试用例

### T1 — 权重与量化

- 同一专家 BF16/INT8/INT4 能独立读取，元数据与 checksum 正确；
- 在线推理过程的量化函数调用计数为 0；
- 进程重启后仅依赖 SSD manifest 可恢复权重；
- 随机选取多个层和专家，解包结果与离线参考一致。

### T2 — 缓存正确性

- 0 字节预算无缓存条目；比容量大的对象不能写入；总容量始终不超限；
- 相同 ID 不同层绝不交叉命中；相同 ID 同层不同 bitwidth 绝不误命中；
- 读取失败、checksum 错误、并发同 key 请求、异步预取失败均可恢复或明确报错；
- GPU 正在执行时不得释放正在使用的专家内存；无死锁。

### T3 — 多层推理

- 替换 2/4 层的 Router Top-k 与原模型一致（或有详细解释、不得默默 PASS）；
- BF16 对照数值误差、INT4/INT8 局部误差、全模型 logits 误差都被记录；
- 有/无 KV Cache 的一致性 smoke test；
- 真实 decode 可完成指定 token 数，且生成长度和 finish reason 合理。

### T4 — SSD 真实性

- 清除应用缓存后，必须触发正确的 SSD 数据读取路径；
- 对设备层 cold-read 的声明必须同时有可辨识 block I/O 证据；若只有 Page Cache 命中，测试标注 `buffered`；
- RAM 预算很小时，系统没有依赖隐藏的专家 BF16 副本来完成全部请求；
- 记录系统 RSS/anon 和 GPU allocated/reserved 以验证内存所有权。

### T5 — 预取

- static 策略不读取未来 Router 结果；oracle 正确标注上界；
- prefetch I/O 计入总读取量；错误预测有浪费字节数；
- 使用前等待 CUDA Event，输出与无预取一致；
- 超预算时回退、不会线程泄漏或无界排队。

### T6 — 报告一致性

- 如果把 CSV 中 `12.20` 改成 `11.00`，自动报告必须同步变化；
- 如果摘要手工写成 `12.8+` 或 `93%+` 与原始数据不一致，验证器必须 FAIL；
- 32 prompt × 3 repeat 等宣称由实际 unique IDs/记录计数推导；
- p95/p99 来自全部真实 decode token，而不是 run 级均值的分位数；
- 所有 FAIL/NOT VERIFIED 必须出现在最终汇总表。

---

## 13. 资源与失败处理

### 13.1 RTX 4060 8GB 显存

- 每次运行前测量**实际剩余**显存；应用缓存容量应留出模型 backbone、attention、KV Cache、解包/反量化、CUDA allocator staging 缓冲区。
- 遇到 OOM：先减 context、batch、prefetch depth、GPU cache budget；必要时先测 L2 而不是 L4。记录新的实际参数、OOM 时的 PyTorch memory summary。
- 可以让热专家放在 RAM，冷专家在 SSD；不要把某个专家“只在 CPU/RAM 计算”未经说明混入 GPU 吞吐对照。

### 13.2 48GB RAM

- 系统 Page Cache 与 Python 进程匿名内存要区分；不要故意把主机推向交换风暴。
- 记录 `vmstat 1` / `free` / `ps` / 进程 smaps 采样，避免把正常缓存页当作隐藏独立拷贝。
- RAM 预算是 **expert cache 的逻辑预算**，需要同时报告模型其他参数占用与进程实际峰值。

### 13.3 NVMe SSD

- 按磁盘剩余空间决定导出层数；不要重复下载多份相同模型权重；支持已有 store 的增量构建。
- 默认不执行 root 级 cache 清理、不写入系统关键路径、不做长时间破坏性磁盘压力测试。
- 若 SSD 冷读与设备性能出现异常，报告读写负载、系统 Page Cache 和温度的可能影响；不假定 SSD 标称带宽等于实际 3 MB 读取性能。

### 13.4 最小完成标准

如果资源不足，仍可完成：G0、G1(L2)、G2(L2)、G3、G4(L2 sync)、G7，并把 G5/G6(full) 标为 `NOT RUN - RESOURCE LIMIT`。不能用简单模拟代替真实模型的 G4 PASS。

---

## 14. 各阶段门禁与停止条件

| Gate | 必须通过的证据 | 阻塞条件 |
|---|---|---|
| G0 | SHA + 环境冻结；历史摘要不一致能被校验器抓到 | 原始报告与数据无法关联；测试失败无诊断 |
| G1 | L2/L4 BF16 数值回归、真实增量 Decode | 路由/专家键错误、NaN、KV Cache 失配、OOM 无回退 |
| G2 | 多精度 store 进程重启后完整读取并校验 | 在线量化、错误版本、文件不一致 |
| G3 | 严格缓存预算、完整事件、并发安全 | 越界、误命中、死锁、吞异常 |
| G4 | 真实 MoE 中 SSD-backed load；物理冷读单独验证 | 只能证明从常驻 RAM/系统缓存读取却声称物理 SSD |
| G5 | 无未来信息泄露的异步预取、正确 I/O 计量 | 使用后续真实路由做可部署策略、输出错误 |
| G6 | 固定/动态策略有成对质量—延迟—I/O 数据 | 模式间缓存污染、不同 prompt 设置、无法解释 precision 执行分布 |
| G7 | 数据驱动报告、自动测试、明确 PASS/FAIL/NOT VERIFIED | 报告硬编码指标、缺原始 token 日志 |

**规则**：G0/G1/G2/G3/G4 中任一失败，不得宣布“多层 SSD 动态卸载已验证”；G5 未完成仍可报告“同步 SSD 卸载已验证，但预取未验证”。G6 没有发现收益不能改写成收益。

---

## 15. 推荐给 Agent 的执行顺序

1. 克隆/切换 `main`，记录 SHA；创建独立 P5b 分支并审计当前文件和实验入口。
2. 复核 P5a.3 数据，生成 `P5A3_CORRECTIONS.md`，修复可复用的自动报告工具。
3. 实现/复用多层替换：先 Layer 0，再 `[0,4]`，最后 `[0,4,8,12]`；跑 BF16 数值回归。
4. 离线建立多精度专家 SSD store 和 manifest；校验随机读取和进程重启恢复。
5. 严格实现缓存预算、层/专家/精度键、事件日志；通过单元与并发测试。
6. 实现同步 SSD 专家卸载；**证明没有依赖被替换层的隐藏完整 BF16 副本**；在真实 Decode 里观察数据路径。
7. 跑固定 INT4/INT8/S4 的 L2/L4 对照，先测 4、8 prompt，再测可行的 32 prompt。
8. 加入 static/oracle 两类预取，正确记录物理/逻辑 I/O 与预取浪费；实现真正的流间同步。
9. 形成质量—延迟—SSD 流量的 Pareto 对照，计算每个 token 的尾延迟。
10. 生成数据驱动的 `docs/p5b/P5B_REPORT.md`，附 `stage_gates.json`、配置、原始数据路径、限制和复现实验命令。
11. 提交代码、测试、配置、文档和小体积结果至分支；模型二进制与巨大日志应按 `.gitignore` 或 Git LFS/外部路径处理。不得直接强推覆盖 `main`。

---

## 16. 最终交付与报告模板

最终交付：

- `docs/p5b/P5B_REPORT.md`：真实数据驱动总结；
- `docs/p5b/ARCHITECTURE.md`：专家生命周期、SSD→RAM→VRAM 路径、缓存规则、异步依赖图；
- `docs/p5b/P5A3_CORRECTIONS.md`：旧报告与原始数据差异复核；
- `results/p5b/<run_id>/`：可溯源的原始 token/expert 事件、质量数据、I/O 数据及完整配置；
- `results/p5b/overview.csv` 与图表：固定/动态/预取各组的质量、速度、SSD I/O、VRAM/RAM；
- 对应的实现代码、自动测试与必要 README/运行脚本。

`P5B_REPORT.md` 需有下列表格，全部由真实文件填充，不能先写“PASS”占位：

| 问题 | 判定 | 最强证据 | 限制 |
|---|---|---|---|
| 多层真实 MoE 动态精度能执行吗？ | PASS/FAIL | L2/L4 运行和数值对照 | ... |
| 能在**在线不量化**时切换专家精度吗？ | PASS/FAIL | 文件 manifest + 精度事件 | ... |
| 模型真正依赖 SSD 专家 store 吗？ | PASS/FAIL | 模型内存所有权 + load trace | ... |
| 已证明**物理 SSD 冷读**吗？ | PASS/NOT VERIFIED | 进程/设备 I/O 证据 | ... |
| 缓存预算可靠吗？ | PASS/FAIL | hard-bound tests + 峰值 | ... |
| 预取隐藏了加载延迟吗？ | PASS/NO BENEFIT/NOT RUN | 同条件消融 | ... |
| 动态 S4 在等预算下改善 Pareto 吗？ | YES/NO/INCONCLUSIVE | 质量—延迟—流量曲线 | ... |
| 端到端 Decode 性能是否优于固定策略？ | YES/NO | TPOT 分布、tokens/s | ... |

### 最终禁止的叙述

- ❌ “从 SSD 读取了 1GB”——但其实统计的是逻辑 bytes，实际 Linux Page Cache 全命中；
- ❌ “实现了原生 INT4 GPU 推理”——实际上先解包为 BF16 再 matmul；
- ❌ “切换次数 87 表明同一专家多次切换”——实际可能是不同专家的选择变化；
- ❌ “S4 质量提高且速度更快”——仅凭少量 Prompt 和未经成对比较的均值；
- ❌ “P5b 全部通过”——某些 gate 未运行、验证失败或摘要数字和原始数据不一致。

### 最终目标

**先获得一个可信、可复现的多层 SSD 多精度专家卸载原型，再讨论性能优化或扩展到更大 MoE 模型。** 与其宣称不可验证的速度收益，不如清晰回答：哪些专家以何种精度从哪里加载，真正花了多少 SSD I/O、H2D、反量化和计算时间，及其对生成质量的实际影响。
