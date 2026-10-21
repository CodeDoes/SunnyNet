# SunnyNet

**English** · [中文](#中文简介)

An original general-purpose sequence architecture, not derived from Transformer, RNN, Mamba, CNN, or any existing design.

Repository: <https://github.com/SunnyClaw-cx/SunnyNet>

Built for CPU training: O(N) sequence complexity with a fixed-size rolling state, no token-to-token interaction, and no materialized activations across the sequence.

Licensed under Apache License 2.0.

---

## English Overview

### What it is

SunnyNet is built around a single recurrent primitive: a convex combination.

```
s_t = p_t ⊙ s_{t-1} + (1 - p_t) ⊙ u_t ,   p_t ∈ [1-δ_i, 1]
```

Both coefficients are non-negative and sum to one, so `|s_t| ≤ max(|s_{t-1}|, |u_t|)`. The state is bounded by construction and cannot diverge. Each channel's retention floor is tied to a memory span `H_i` laid out on a log scale over `[2, 8192]`, so one layer carries time scales from two tokens to eight thousand.

The design targets the real CPU bottleneck — memory bandwidth and cache behavior — not FLOPs. It avoids dense matmul, keeps the state dimension fixed, and passes only the state across chunks.

### Measured results

All numbers are measured on one machine (i7-8750H, AVX2, PyTorch 2.14.1 CPU) with per-cell process isolation. Full data, including negative results, is in [docs/EXPERIMENT_REPORT.md](docs/EXPERIMENT_REPORT.md).

| Metric | SunnyNet | Best peer | Setting |
|---|---|---|---|
| Training step, ms/token | 0.0953 | RWKV 0.0982 | T=256, batch 8, tied over 9 paired rounds |
| Quality, ppl | 57.83 (2nd of 7) | H3 53.93 | 600 steps, single seed, not converged |
| Decode, ms/step at T=4096 | 2.85 | RWKV 1.56 | B=1, single sequence |
| Forward, ms/token at T=4096 | 0.0352 | Hyena 0.0184 | batch=1, pure forward |

- Training step is statistically tied with RWKV (3–4 of 9 rounds faster, median paired ratio 1.042) and 16–41% faster than H3, Mamba, Hyena, and linear attention. The tie holds only at L=4; at L=6 the same parameter budget is 30–40% slower.
- Long-context decoding: single-step cost stays flat from T=256 to T=4096 (3.28 → 2.85 ms), while a parameter-matched Transformer climbs 2.55 → 6.22 ms. SunnyNet is 2.18× faster at T=4096. Against O(N) peers it is not the fastest — RWKV decodes 1.9–2.6× faster at every length.
- Forward pass: crossover with a matched Transformer at T≈384 (6-layer) or T≈600 (4-layer). At T=4096 it is 22.7× faster per token. Against O(N) peers there is no length where it leads all of them.
- Cross-distribution: on independently collected external data, 1st of 7 on Chinese tech news, 2nd of 7 on English arXiv abstracts.
- Hand-written AVX2 kernel (competitive gating, gate probability, and scan fused into one pass): 1.10–1.17× on the training step, bit-identical loss. Needs a C compiler (w64devkit / GCC).

### Limitations

- Short sequences are a structural weakness: below T≈384 (batch=1 forward) a standard Transformer is faster. The cause is operator count, not FLOPs.
- No 0.5B quality number. On the 6 GB GPU available here, forward and a single SGD step run (5.0 GB peak); AdamW needs 7.6 GB and does not fit.
- The scaling trend is not monotonic: 2nd at 1.4M, 1st at 20M, 2nd at 100M.
- Long-range retrieval is verified only on structured text (code, far-bucket hit 0.13); on natural language it is near zero.
- No GPU kernel. The architecture runs on CUDA unmodified (8.0–15.2× forward on a GTX 1060), but there is no Triton or CUDA implementation.
- Decode speed is not competitive against O(N) peers: RWKV decodes 1.9–2.6× faster at every length tested.

### CPU inference runtime

`sunnynet/runtime/` is a standalone C kernel (AVX2/FMA + OpenMP) that has no Python or PyTorch dependency. It implements the general first-order recurrence

```
s_t = a_t * s_{t-1} + b_t              (FMA)
s_t = a_t * s_{t-1} + (1 - a_t) * u_t  (gated)
```

RWKV's wkv, Mamba/S6 and H3 diagonal SSM, S4, linear-attention KV accumulation, and SunnyNet's rolling state all reduce to one of these two forms. Each is mapped to the kernel and checked elementwise against its own PyTorch recurrence (max diff 1.2e-07 to 1.9e-06). Against torch's vectorized block scan (`gated_scan`, cumsum closed form), the kernel is 9.90× faster at the median (3.56–19.29×, single thread, B=1, per-cell process isolation).

The convex-combination state has a bounded magnitude by construction. Over T=2048 the state magnitude drifts 1.06× versus 7.94× for unbounded accumulation, and under a fixed INT8 scale the relative error is 4.0e-02 versus 5.0e-01.

The whole decode step — embedding lookup, per-layer norm, depthwise conv history, gated competition, state recurrence, readout, final norm, LM head — also runs in C with no intermediate tensors. Weights are exported from a checkpoint into a packed fp32 blob (matmul weights pre-transposed for GEMV, `circ ∘ W_s` folded into one matrix). Against `SunnyNet.forward_step`, per-token logits agree to 1.24e-05 and argmax matches 48/48. Whole-model decode is 1.53× / 2.05× / 1.97× faster than torch at prefill T=256/1024/2048 (B=1, single thread, 3 rounds, C wins 3/3 every cell).

CLI: `python -m sunnynet.cli --ckpt runs/small.pt --prompt "..."` (`--backend c` for the pure-C decode path).

Foreign weights: `sunnynet/runtime/gguf.py` reads GGUF and `rwkv7.py` runs RWKV-7 per token without PyTorch. Dequantized tensors are bit-identical to the official `.pth` (max diff 0), and per-token logits match an independent reference implementation to 4.79e-06 (argmax 12/12). The official RWKV-7 191M model generates coherent text (English, Chinese, and Python all verified). This path is numpy, not C — it demonstrates that the runtime consumes external weights correctly, and its ~388 ms/step is not a speed result. RWKV-7's state is a per-head matrix with a rank-1 removal term, so a matrix-valued recurrence layer is needed in the IR beyond the elementwise forms above.

---

## 中文简介

原创通用序列神经网络架构，不基于 Transformer、RNN、Mamba、CNN 等任何现有架构。

仓库地址：<https://github.com/SunnyClaw-cx/SunnyNet>

架构面向 CPU 训练设计，O(N) 线性序列复杂度，降低内存带宽消耗，提升 CPU 缓存利用率。
支持通用文本自回归预训练，可进行指令微调与领域微调。

源代码使用 Apache License 2.0 开源。

---

## 目录

- [快速开始](#快速开始)
- [架构核心思想](#架构核心思想)
- [原创组件](#原创组件)
- [实测结果](#实测结果)
- [与六家主流架构的对照](#与六家主流架构的对照)
- [线性架构的 CPU 推理 Runtime](#线性架构的-cpu-推理-runtime)
- [仓库结构](#仓库结构)
- [原创性说明](#原创性说明)
- [风险与局限](#风险与局限)
- [赞助](#赞助)
- [许可证](#许可证)

---

## 快速开始

### 环境

- Python ≥ 3.10
- PyTorch ≥ 2.1（CPU 版即可，无 GPU 依赖）
- numpy

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy matplotlib pytest
```

### 1. 下载语料并构建 token 流

```bash
# 采集真实语料: Project Gutenberg 中文古籍/古典文学 + 英文文学 + TheAlgorithms/Python 源码
python prepare_data.py

# 训练 BPE 分词器 -> 编码为 token 流 -> 划分训练/验证
python build_corpus.py --vocab-size 8192
```

验证集按文件级切分，同一文本的相邻片段不会跨训练与验证。

### 2. 训练

```bash
# 小模型快速验证（4 层）
python train/train.py --preset small --steps 800 --batch-size 8 --seq-len 256

# 20M 原型
python train/train.py --preset proto20m --steps 5000 --batch-size 4 --seq-len 512
```

### 3. 评估

```bash
python train/evaluate.py --ckpt runs/<你的>.pt --gen \
    --out experiments/results/eval_sunnynet.json
```

### 4. 完整实验

```bash
bash experiments/battery.sh          # 基线 + 消融 + 种子 + 缩放
bash experiments/battery2.sh         # 20M 缩放对照 + 长序列训练对照

# O(N) 验证: 前向延迟 vs 序列长度
python experiments/long_sequence.py --only latency

python experiments/make_figures.py
python experiments/collect_results.py --out docs/_tables.md
```

手写 AVX2 内核需 C 编译器（w64devkit / GCC）：
将 `SUNNY_W64` 指向 w64devkit 根目录，或把 `gcc` 加入 PATH。

```bash
export SUNNY_W64=<w64devkit 解压目录>
python -X utf8 experiments/_kernel_bench.py        # 算子级加速比
python -X utf8 experiments/_core_e2e.py 256 6 8    # 融合内核训练步 A/B
```

---

## 架构核心思想

CPU 预训练的瓶颈通常不在浮点算力，而在内存带宽与缓存命中率。SunnyNet 的设计围绕三条硬约束：

| 约束 | 对应设计 |
|---|---|
| 序列复杂度严格 O(N) | 滚动状态 + 一阶线性递推，无任何 token 两两交互 |
| 不保存整条序列的中间激活 | 状态维度固定，跨块只传状态；反向按需求重算 |
| 规避巨型稠密矩阵乘法 | 低秩分解 + 深度可分卷积 + 三对角循环滤波 |

核心状态更新是凸组合：

```
s_t = p_t ⊙ s_{t-1} + (1 - p_t) ⊙ u_t ,     p_t ∈ [1-δ_i, 1]
```

两个系数非负且和为 1，因此 `|s_t| ≤ max(|s_{t-1}|, |u_t|)`，状态天然有界，不会发散。

保留率 `p_t` 的逐通道下界由记忆跨度 `H_i` 决定，`H_i` 在 log 刻度上遍历 `[2, 8192]`，
同一层内同时存在从"只记 2 个 token"到"记忆 8192 个 token"的全时间尺度，
兼顾局部语法与跨段落、跨代码块的长距离依赖。

完整数学推导见 **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**。

---

## 原创组件

| 组件 | 方案 | 关键性质 |
|---|---|---|
| 状态更新 | 多时间尺度凸组合滚动 | 状态有界，不发散；多尺度输出门控 |
| 非线性激活 | SoftWedge `f(z)=z+a·s(z)`，底式 `s(z)=½+½·z/(1+\|z\|)` | 导数 `f' ∈ (1, 1+a/2]`，处处非零，无梯度断崖 |
| 归一化 | CenteredScaleNorm | 分母设平滑下界 `s_min`，近似常量向量不被放大（深堆叠前提） |
| 初始化 | 谱锚定（QR 正交切片，σ_max 直接锚定） | 条件数为 1，前向增益谱平坦；ρ=1.0 由实测标定 |
| 跨层传递 | 状态接力 + 特征残差流 | 状态尺度不随深度放大；残差直连兜住深度梯度（实测 −28% ppl） |
| 通道竞争 | 组内成对竞争门控 | 与 softmax 同语义，但 Jacobian 为 O(1) 而非 O(1/d) |
| 并行扫描 | 对数域闭式解 | 与逐步递推代数等价；块首状态可由块末精确反解 |

未复用 ReLU / GELU / SiLU / Mish、LayerNorm / RMSNorm / InstanceNorm / GroupNorm、
Xavier / Glorot / He / MSRA 初始化、LSTM / GRU 门控公式、SSM / HiPPO 离散化。

---

## 实测结果

全部实验共用同一 token 流、同一配置、同一 token 预算与线程数。参数量用 meta device
实例化对齐，层数偏离基准不超过 ±1。完整数据见 [docs/EXPERIMENT_REPORT.md](docs/EXPERIMENT_REPORT.md)。

### 训练速度

- **同架构提速**：算子融合 19–41%；减层加宽 12–17%；窄残差流 + `p.clamp` 判掉 17.7–20.7%。
- **对同复杂度同行**：T=256 训练步 0.0953 ms/token，名义快 rwkv 3.0%；9 轮逐轮配对检验
  不显著（3–4/9 轮占优）⇒ 写作"与 rwkv 统计持平"，不写"超越"。对 h3/mamba/hyena/linear_attn
  稳定领先 16–41%。
- **持平的边界**：仅 `L=4` 下成立；层数对齐到 `L=6` 时同预算窄配置反慢 30–40%；T≥1024
  同行优势仍在。
- **手写内核**：融合竞争门控 + 门控概率 + 扫描，训练步 A/B 三指标一致同向，
  T=256 6/6 占优、比值 0.8746（1.143×）；ppl 与 torch 逐位一致（不改变质量，只提速度）。
  推理态内核收益不显著（B=1 张量太小，跨语言调用开销与单算子同量级），默认关闭。

### 推理

- 单步解码成本随上下文：SunnyNet 平坦（O(1)，2.87→2.90 ms，漂移 ±5%），
  Transformer 单调上升（O(t)，2.49→6.85 ms）。交叉点 T≈512，T=4096 时 SunnyNet 快 **2.36×**，
  差距随 T 继续拉大。不需内核即可获得。
- 短上下文（T<512）解码不是优势场景，内核融合在此也无收益。
- 边界：单条序列、B=1、随机初始化模型、2 轮中位，未测批处理解码，不含 KV cache 量化/分页。
- 与 O(N) 同行同场对照见下节：SunnyNet 解码**不是最快**，rwkv 全程第 1、快 1.9–2.6×。

### 质量

- 架构修正（残差 + CSN 地板）后，T=256 ppl 由 93.82 降至 57.83，七架构第二
  （仅落后 H3 53.93）；其余为 Mamba 63.34 / RWKV 64.10 / Hyena 82.50 / Transformer 90.05 / 线性注意力 91.91。
- 20M 减层加宽后 ppl 再降 10.9%（54.82），在同协议对照中升至五架构第一。
- **scaling 趋势非单调**：small 质量第二 → 20M 质量第一 → 100M 退回第二（109.47，落后
  rwkv 105.66 约 3.6%）。不能说"规模越大优势越大"。

### 长序列与内存

- **前向**：交叉点 T≈384（6 层对照）/ T≈600（4 层对照）；T=4096 时每 token 比
  Transformer 快 22.7×（batch=1、纯前向、8 线程、1.4M 小模型）。
- **训练态 T=1024**：SunnyNet 速度第 1（0.1319 ms/tok），峰值内存 1812 MB 第 2 低档
  （Transformer 3192 MB，高 76%）。
- **对 O(N) 同行的前向**：无稳定领先区间，随 T 从第 6 名追到第 3 名，全程未登顶；
  训练态排名不同（T=1024 时第 1）。

### 跨分布泛化

外部独立数据集（原三域训练，新域测试）：中文科技新闻 ppl 321.08 第 1/7（该域未登录字符率
23%）、英文 arxiv 摘要 35.58 第 2/7；OOD 落差 1.480，七架构最低档（与 hyena 并列）。
绝对 ppl 仍高、600 步未收敛、单种子，只能说"跨分布不比同复杂度同行差"。

### 短序列

交叉点 T≈384 只在 batch=1 纯前向成立。训练态（batch=8）交叉点下移到 T<256：
T=256/384 时比 Transformer 快 1.5–2.4×。T=128 仍慢 10.4%（旧 d128）/ 4.5%（窄配置
smalln80）——窄残差流是已验证的缓解手段。

### GPU

架构无需 CUDA 特化改写即可在 CUDA 上运行。B=8 加速：前向 8.0–15.2×、训练步 9.9–12.5×
（未做 CUDA 侧优化，是下界）。本机 GPU（GTX 1060 6GB）仅用于正确性验证与加速比测量，
无同行 GPU 对照。

### 0.5B 在 6GB 卡上的实测边界

| 档位 | 峰值显存 | 结果 |
|---|---|---|
| 构建（498M 参数，fp32） | 1902 MB | 通过 |
| 前向 T=512 | 2010 MB | 通过 |
| 前向 T=1024 | 2060 MB | 通过 |
| **SGD 单步**（T=512） | **5004 MB** | **通过**，但只剩 563 MB（9% 余量） |
| AdamW 单步 | >6144 MB | **OOM** |

约束来自优化器状态的算术：fp32 参数 1.9GB + 梯度 1.9GB + AdamW 一阶/二阶矩 3.8GB
= 7.6GB > 6GB，SGD 只有梯度一项所以勉强塞得下。
优化器状态 CPU offload 在本机失败——主机物理内存只剩 4.5GB，装不下 offload 的
3.7GB 优化器态，触发页文件压力并拖垮 CUDA 上下文。
⇒ **0.5B 在这台机器上可跑单步 SGD，但无多步收敛与质量数据。**

测量注意：同一进程内先跑 T=512/T=1024 前向再跑训练步，会因显存碎片使训练步 OOM
（实测剩余 1.9GB 仍分配不出 16MB）；**每档必须独立子进程**，否则得到错误的"不可行"结论。

---

## 与六家主流架构的对照

对照对象：`h3`、`rwkv`、`mamba`、`hyena`、`linear_attn`、`transformer`。
所有数字来自本机实测、逐格独立子进程。完整数据见 [docs/EXPERIMENT_REPORT.md](docs/EXPERIMENT_REPORT.md) 第 23 节。

### 记分卡

| 维度 | SunnyNet 位置 | 最优者 | 差距 |
|---|---|---|---|
| 质量 ppl（1.4M / 600 步） | 第 2（57.83） | h3 53.93 | 落后 7.2% |
| 质量 ppl（20M / 200 步） | **第 1**（77.90） | sunnynet | — |
| 质量 ppl（100M / 100 步） | 第 2（109.47） | rwkv 105.66 | 落后 3.6% |
| 训练速度（L=4 等栈参数） | 与 rwkv 统计持平 | rwkv（名义） | 配对检验不显著 |
| 训练速度（L=6 层数对齐） | **落后** | rwkv | 慢 30–40% |
| 训练态内存（T=1024） | 第 2 低 | rwkv 1611MB | 高 12%，比 TF 低 43% |
| 前向 ms/token（batch=1） | 第 6 → 第 3（随 T） | hyena | 全程未登顶 |
| **解码 ms/step（B=1）** | 第 5 → 第 2 | **rwkv** | 慢 1.9–2.6× |
| 长距离检索（代码域） | 与 TF 持平 | 并列 | — |
| 长距离检索（自然语言） | 接近 0 | — | 无优势 |
| 跨分布泛化 | 中文新域第 1 / 英文第 2 | sunnynet | — |

### 解码态七架构对照

口径：每个 (架构, T) 独立子进程；先逐 token 走满 T 步建立真实解码状态（不计时），
再测该长度下单步解码 ms；1 线程；2 轮取中位。

ms/step（越低越好）：

| 架构 | T=256 | T=512 | T=1024 | T=2048 | T=4096 |
|---|---|---|---|---|---|
| `rwkv` | **1.768** | **1.574** | **1.598** | **2.113** | **1.564** |
| `h3` | 2.517 | 2.524 | 2.487 | 2.969 | 3.227 |
| `transformer` | 2.550 | 2.667 | 3.426 | 5.049 | 6.219 |
| `mamba` | 2.686 | 2.742 | 2.573 | 2.773 | 3.205 |
| **`sunnynet`** | **3.279** | **4.067** | **3.086** | **3.331** | **2.854** |
| `hyena` | 3.346 | 3.583 | 3.662 | 5.101 | 4.530 |
| `linear_attn` | 4.061 | 3.452 | 4.599 | 3.676 | 3.624 |

读数：

- Transformer 的 O(t) 被复现（2.550 → 6.219 ms），T=4096 时 SunnyNet 快 2.18×。
- SunnyNet 单步成本大致平坦，排名随 T 改善（5→7→4→4→2），但**从未追上 rwkv**。
- **解码态最优是 rwkv**（1.57–2.11 ms），单步只有一条 `p*s + k*v` 逐元素链；
  SunnyNet 单步含竞争门控（3 次 sigmoid）、门控概率、扫描、反馈支路，算子数是它的数倍。
- 五个 O(N) 同行此前都无单步解码接口，本仓库为其实现 `init_state`/`step`，
  与整段前向数值一致性 maxdiff ≤ 4e-05。

### 差距的共同根因

落后的维度（解码速度、前向速度、L=6 训练速度、100M 质量、自然语言长距离检索）
收敛到同一个根因：**逐元素算子数量**。证据：训练步 4524 次 aten 调用；
减层加宽（等栈参数）训练快 12–17% 且 ppl −10.9%；层数对齐到 L=6 后同预算窄配置反慢 30–40%；
解码态 rwkv 靠单条逐元素链取胜。⇒ 有效方向只有减算子数，不是加宽、调 batch 或换矩阵库。

---

## 线性架构的 CPU 推理 Runtime

`sunnynet/runtime/` 是独立于 PyTorch 的 C 推理内核（AVX2/FMA + OpenMP），只依赖 ctypes
与 C 编译器。它实现通用一阶递推

```
s_t = a_t * s_{t-1} + b_t              (FMA)
s_t = a_t * s_{t-1} + (1 - a_t) * u_t  (门控)
```

RWKV 的 wkv、Mamba/S6 与 H3 的对角 SSM、S4、线性注意力的 KV 累加、SunnyNet 的滚动状态，
全部归约为这两种形式之一。

### 跨架构映射（`experiments/_unified_ir_map.py`）

各架构递推映射到内核参数，与自身 PyTorch 递推逐元素比对，B=2/T=128/N=128/fp32：

| 架构递推形式 | 映射 | max\|diff\| |
|---|---|---|
| RWKV-4 wkv | `a=p`, `b=k*v` | 9.5e-07 |
| Mamba / S6 SSM | `a=decay`, `u=bu` | 2.4e-07 |
| H3 对角 SSM | `a=decay`, `u=v` | 1.2e-07 |
| 线性注意力累积 KV | `a=1`, `b=k v^T` | 0.0 |
| SunnyNet 滚动状态 | `a=p`, `u=u` | 1.9e-06 |
| S4 对角 SSM | `a=A_bar`, `b=B_bar x` | 7.2e-07 |

Hyena 是长卷积（FIR）递推，不属一阶状态递推，需独立卷积后端。

### 速度（`experiments/_runtime_bench.py`）

每格独立子进程、单线程、B=1、10 次中位。基线是 torch 向量化分块扫描
（`ops.gated_scan`，cumsum 闭式解），不是 Python 循环。

| T | N | 模式 | 向量化 ms | 内核 ms | 倍数 |
|---|---|---|---|---|---|
| 256 | 128 | FMA | 0.391 | 0.043 | 9.08× |
| 1024 | 128 | FMA | 1.256 | 0.132 | 9.54× |
| 4096 | 128 | GATED | 8.500 | 0.806 | 10.55× |
| 4096 | 640 | FMA | 26.944 | 7.574 | 3.56× |

中位 **9.90×**，范围 3.56–19.29×。N=640 时降到 3.6–4.0×（单线程内存带宽限制）。
与内核的三分支最大数值差 7.63e-06。

**边界**：这是**批量跨段扫描**口径（B=1、整段 T）。接入 B=1 单步解码后收益消失
（四格比值 1.023 / 1.072 / 0.897 / 1.073，方向不一致，8 线程 T=256 反慢 10.3%），
⇒ 不得声称端到端解码加速。原因：模型侧 T=1、B=1，跨语言调用与状态搬运成本
与单算子同量级。B=1 单序列也不是本 Runtime 的目标场景（解码态最优是 rwkv）。

### 整模型 C 解码（`sunnynet/runtime/sunnynet.c`）

单步算子内核之上，整个前向也搬进了 C：嵌入查表、逐层 norm / 深度卷积历史 / 门控竞争 /
状态递推 / 读出、终态归一、LM head，全在一条无中间张量的路径上完成。权重由
`experiments/_export_runtime.py` 从 ckpt 导出为打包 fp32 blob（矩阵乘权重预转置为
`(in, out)` 以配合 GEMV，`circ ∘ W_s` 折成单矩阵）。

数值一致性（`experiments/_runtime_model_equiv.py`，1.4M ckpt，48 token 逐步）：

| 口径 | 结果 |
|---|---|
| 单步 logits max\|diff\| | 1.24e-05 |
| argmax 一致 | 48 / 48 |
| prefill 末 token 与 torch 整段前向 | 3.81e-06 |

端到端解码延迟（`experiments/_runtime_decode.py`，B=1 单序列、单线程、每格独立子进程、
先走满 prefill 再测单步、3 轮交错）：

| prefill T | torch ms/step | C ms/step | 比值 | 逐轮占优 |
|---|---|---|---|---|
| 256 | 3.796 | 2.487 | 1.53× | 3/3 |
| 1024 | 3.802 | 1.851 | 2.05× | 3/3 |
| 2048 | 4.309 | 2.189 | 1.97× | 3/3 |

中位比值与逐轮符号检验一致（四格全 3/3），可判定为真实提升，非轮间噪声。
提速来源不是算子融合，而是消灭每步约 1100 次 aten 调用 / 21 次 GEMV / 20 次 cat
（torch 单步剖析：`aten::mm` 16.4%、`aten::mul` 10.2%、`aten::select` 8.5%、`aten::cat` 8.0%）。

**口径边界**：上表是**单线程 B=1**。CLI 默认 8 线程时 C 后端 4.40 ms/tok vs
torch 4.56 ms/tok（24 token 单次读数，未做多轮），差距被线程调度摊平——B=1 小张量
并行是负收益（§20.2 已记录 1 线程比 8 线程快 1.42×）。C 后端的优势在单线程与更大批量，
多线程 B=1 下不要按上表线性外推。

### 状态量化（`experiments/_state_quant.py`）

凸组合状态幅度由构造有界。T=2048、真实 ckpt 第 0 层状态：

| 模式 | \|s\| 漂移 (末/1/8 处) | 固定标度相对误差 | 溢出通道比 |
|---|---|---|---|
| 有界递推 | 1.06× | 4.0e-02 | 0.78% |
| 无界累加 | 7.94× | 5.0e-01 | 5.47% |

固定标度下的相对误差相差 12.5×。
边界：只测第 0 层、单 ckpt（1.4M）、单条随机序列，不做端到端 ppl 对照。

### GGUF 只读（`sunnynet/runtime/gguf.py` + `rwkv7.py`）

读入外部 GGUF 权重并逐 token 自回归生成，不经 PyTorch。首个打通的模型是官方
RWKV-7 191M（`BlinkDL/rwkv-7-world` 的 `.pth` 经 llama.cpp 转 GGUF，bf16）。
张量命名与递推形式取自 llama.cpp 的 `src/models/rwkv7*.cpp`；参考数学取自
BlinkDL/RWKV-LM 的 `RWKV-v7/rwkv_v7_numpy.py`。

数值验证（`experiments/_rwkv7_gguf_equiv.py`）：GGUF 反量化结果与官方 `.pth`
**逐位一致**（`max|diff| = 0`）；逐 token 单步 logits 与独立参考实现：

| 口径 | 结果 |
|---|---|
| 单步 logits max\|diff\| | 4.79e-06（bf16 精度内） |
| argmax 一致 | 12 / 12 |

生成样例（`experiments/_rwkv7_gguf_bench.py`，温度 0.7 / top-k 20 / seed 42）：

```
The Eiffel Tower is located in the city of
-> Paris, France. The Eiffel Tower was designed by Gustave Eiffel in 1887 as a monument to
def fibonacci(n):
-> \n    if n < 2:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)
```

**口径边界**：`rwkv7.py` 目前是 **numpy 参考路径**，不是 C 内核。它证明的是
"Runtime 能读外部 GGUF 权重并算对"，解码约 388 ms/step——**这个数字不作速度成果陈述**。
把 RWKV-7 的递推搬进 C 内核（复用 `recur.c` 的向量化骨架）是下一步。

一条与既有叙事的差异必须记录：RWKV-7 的状态是**逐 head 的矩阵**
`S ∈ R^{64×64}`，更新为 `S_t = S_{t-1}⊙w_t − (S_{t-1}k_t)k_tᵀ + v_t k_tᵀ`，
含 rank-1 去除项——**不是**元素级 `s = a⊙s + b`。统一 Runtime 若要覆盖它，
IR 需要在元素级递推之外补一个矩阵值递推（外积 / rank-1 校正）层级。

### 命令行

```bash
# torch 后端
python -m sunnynet.cli --ckpt experiments/ckpts_ood/sunnynet.pt \
    --prompt "子曰：" --max-new 80 --stats

# 纯 C 后端 (先导出权重 blob, 再解码)
python -X utf8 experiments/_export_runtime.py \
    experiments/ckpts_ood/sunnynet.pt experiments/ckpts_ood/sunnynet.runtime.bin
python -m sunnynet.cli --ckpt experiments/ckpts_ood/sunnynet.pt \
    --backend c --prompt "子曰：" --max-new 80 --stats

# 外部 GGUF 权重 (RWKV-7)
python -m sunnynet.cli --gguf experiments/gguf/rwkv7-191M-world-bf16.gguf \
    --prompt "The Eiffel Tower is located in the city of" --max-new 40
```

### 路线

现有资产：原创架构、AVX2 融合内核（训练步 1.10–1.17×）、通用递推 C Runtime
（对向量化基线中位 9.9×）、**C 整模型单步解码（B=1 单线程 1.53–2.05×）**、
跨架构映射验证、**GGUF 只读 + RWKV-7 权重跑通（数值与官方 .pth 等价）**。

近期补齐：RWKV-7 递推的 C 内核（把 388 ms/step 的 numpy 路径换成向量化单步）、
INT8 权重量化、更大批量的 C 解码（多序列共享权重读取）。

中期：把内核从 SunnyNet 定制扩展为任意 `s_t = a_t*s_{t-1} + b_t` 的通用单步/批量扫描，
并补矩阵值递推（外积 / rank-1 校正）覆盖 RWKV-7、DeltaNet 这类；补 AVX-512 与 NEON
后端；OpenAI 兼容 HTTP 服务。

不做的：复刻 llama.cpp 的通用张量算子层（GGML 为 Transformer 设计，逐算子调度）；
打平 Transformers / vLLM（分别为模型仓库与 GPU 服务赛道）。

---

## 仓库结构

```
sunnynet/
  ops.py          原创算子: poly_sigmoid / softwedge / CenteredScaleNorm /
                  多尺度衰减 / 对数域并行扫描 / 通用门控扫描
  layers.py       SunnyBlock: 状态更新单元 + 循环滤波 + 局部卷积 + 非线性支路
  kernels/        AVX2 手写内核 (scan / 融合 core / 推理态单步) + 自动编译加载
  linear_baselines.py  O(N) 同复杂度对照: 线性注意力 / RWKV / Mamba / Hyena / H3
  model.py        SunnyNet 主体 (状态 (B,L,D), 逐层接力)
  init.py         谱锚定初始化
  cli.py          命令行推理 (加载 ckpt -> 分词器 -> 自回归解码, --backend torch/c;
                  --gguf 走外部 RWKV-7 权重)
  runtime/        独立 C 推理 Runtime
    recur.c         通用一阶递推内核 (扫描 / 反向 / 单步 / INT8 量化)
    sunnynet.c      整模型单步解码 (B=1, 无中间张量)
    model.py        ctypes 绑定
    gguf.py         GGUF 只读加载 (f16/bf16/f32 + 量化反量化)
    rwkv7.py        RWKV-7 单步递推 (numpy 参考路径, 读 GGUF 权重)
  baseline.py     同参数量 Transformer 对照 (仅用于对照, 非架构本体)
  tokenizer.py    从零实现的 BPE 分词器
  profiling.py    真实性能探针 (峰值内存 / 访存 / 吞吐)
train/
  train.py        统一训练器 (两种架构共用同一循环)
  evaluate.py     困惑度 / 长距离探针 / 生成样例
experiments/
  run_experiments.py   实验编排 (baseline / ablation / seeds / scale / probe)
  battery.sh           批量运行基线 + 消融 + 种子 + 缩放
  battery2.sh          20M 缩放对照 + 长序列训练对照
  long_sequence.py     前向延迟 vs 序列长度 (O(N) 验证) + 长序列训练
  make_figures.py      生成全部图表 -> docs/figures/
  collect_results.py   汇总所有结果为 Markdown 表
tests/
  test_architecture.py  架构自检 (分块一致性 / 因果性 / 梯度界 / 深度稳定, 63 项)
  test_runtime_model.py C 整模型解码一致性
  test_rwkv7_gguf.py    GGUF 加载 / 张量对齐 / RWKV-7 递推等价 / 分词往返
  test_smoke_pipeline.py 端到端冒烟 (合成语料 -> 训练 -> 评估)
prepare_data.py   真实语料采集
build_corpus.py   分词器训练 + token 流构建
docs/
  ARCHITECTURE.md  完整数学推导
  EXPERIMENT_REPORT.md 实测实验报告 (含负面结论)
  figures/         实验图表
```

---

## 原创性说明

SunnyNet 为独立原创架构，同时有两点需要说明：

- 其状态递推在数学形式上与线性注意力类工作有亲缘性（都是一阶线性递推），这一点不回避；
- 门控构造（凸组合 + 对数均匀多尺度 + 逐元素保护门）、层间传递（状态接力 + 特征残差流）、
  激活函数、归一化、初始化均为自行设计与实测标定；
- 未参考或移植任何既有项目的代码与结构。

架构设计过程中有四处结论被实测推翻并修正：softmax 通道竞争导致梯度随深度归零；
谱半径层深衰减的符号设反；CenteredScaleNorm 的 `s_min` 高于实际激活尺度导致归一化退化为
常数缩放；原设计排除残差反而造成最大单项质量损失。修正依据与数据记录在
docs/ARCHITECTURE.md 第 3/4/5 节。保留这些记录是为了说明各设计约束的来源。

---

## 风险与局限

1. **短序列劣势**：O(N) 优势只在长序列兑现。batch=1 前向交叉点 T≈384（6 层）/ T≈600（4 层），
   短于此慢于标准 Transformer。该架构不适合短序列场景。
2. **短序列下质量仍不及 H3**：T=256、600 步同协议对比中 ppl 57.83，不及 H3 53.93（差约 7%）。
   该对比未收敛、单种子、学习率未逐架构调优，只能读作早期样本效率排序。
3. **长序列对同行也无绝对优势**：T=4096 时 Hyena 仍快 1.9×、H3 基本追平；不存在"领先全部
   同行"的长度区间。训练态 T=1024 下速度第 1、内存第 2 低档——前向与训练态排名不同，引用须带条件。
4. **早期耗时数据失真**：未做进程隔离时同进程累积构建会使读数偏高达 2 倍以上，且放大倍数随架构而变。
5. **长距离依赖**：滚动状态的信息容量受限于固定维度，理论上界低于全注意力。大跨度通道只保证
   "不被遗忘"，不保证"被检索"。
6. **伪收敛**：loss / ppl 下降不等于具备推理与指令跟随能力。
7. **语料覆盖**：中文维基百科在本机网络不可达，缺少"百科"域，实验只覆盖 古籍 / 英文文学 / 代码 三个真实域。
8. **规模**：已完成 1.4M / 20M / 100M 实测。0.5B 只有前向与单步 SGD 可跑（GPU 6GB 边界态），
   多步 loss 无下降趋势，没有任何 0.5B 质量数据。
9. **消融未成结论**：600 步未收敛，关组件后 ppl 反而更低，不构成组件有效性证据。
10. **同复杂度对比未收敛**：七架构对比仅 600 步，且未逐架构调优学习率，架构间小差距落在种子噪声内。
11. **`torch.compile` 不可用**：本机 MSVC 缺 Windows SDK 头文件，Inductor 的 Windows 分支硬编码
    要求 MSVC 风格命令行；GPU 侧亦无 nvcc / CUDA toolkit / triton。手写内核绕开 Inductor，已独立打通。
12. **GPU 结论的边界**：加速比是未做 CUDA 侧优化的下界；无同行 GPU 对照，不构成"GPU 上性能竞争力"的证据。

详见 **[docs/EXPERIMENT_REPORT.md](docs/EXPERIMENT_REPORT.md)** 与 **[safety_notice.md](safety_notice.md)**。
模型为实验科研项目，中医相关输出不可作为医疗诊断依据。

---

## 赞助

<a href="https://afdian.com/a/sunny-cx">
  <img src="https://img.shields.io/badge/爱发电-赞助支持-946CE6?style=for-the-badge" alt="爱发电">
</a>

---

## 许可证

[Apache License 2.0](LICENSE)

Copyright 2026 sunny-cx