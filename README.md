# SunnyNet

**SunnyNet：全新原创通用序列神经网络架构，不基于 Transformer、RNN、Mamba、CNN 等任何现有神经网络架构。**

仓库地址：<https://github.com/SunnyClaw-cx/SunnyNet>

架构专为 CPU 高效训练设计，O(N) 线性序列复杂度，降低内存带宽消耗，提升 CPU 缓存利用率。

支持通用文本自回归预训练，可进行指令微调、领域微调（包含中医古籍领域）。

源代码使用 Apache License 2.0 开源。

---

## 目录

- [快速开始](#快速开始)
- [架构核心思想](#架构核心思想)
- [原创组件](#原创组件)
- [附带实验](#附带实验)
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

> 验证集按**文件级**切分，同一文本的相邻片段不会同时出现在训练与验证中，杜绝信息泄漏。

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

### 4. 完整实验（基线对照 / 消融 / 种子稳定性 / 缩放 / 长序列）

```bash
# 一次性跑完前四组 (基线 1200 步 + 消融/种子 600 步 + 20M 缩放)
bash experiments/battery.sh

# 20M 缩放对照 + 长序列训练对照
bash experiments/battery2.sh

# O(N) 验证: 前向延迟 vs 序列长度 (核心实验)
python experiments/long_sequence.py --only latency

# 出图与汇总
python experiments/make_figures.py
python experiments/collect_results.py --out docs/_tables.md
```

---

## 架构核心思想

在 CPU 上做预训练，瓶颈通常不在浮点算力，而在**内存带宽**与**缓存命中率**。
SunnyNet 的全部设计围绕三条硬约束展开：

| 约束 | 对应设计 |
|---|---|
| 序列复杂度严格 O(N) | 滚动状态 + 一阶线性递推，无任何 token 两两交互 |
| 不保存整条序列的中间激活 | 状态维度固定，跨块只传状态；反向按需求重算 |
| 规避巨型稠密矩阵乘法 | 低秩分解 + 深度可分卷积 + 三对角循环滤波 |

核心状态更新是**凸组合**：

```
s_t = p_t ⊙ s_{t-1} + (1 - p_t) ⊙ u_t ,     p_t ∈ [1-δ_i, 1]
```

两个系数非负且和为 1，因此 `|s_t| ≤ max(|s_{t-1}|, |u_t|)` —— **状态天然有界，不会发散**。

保留率 `p_t` 的逐通道下界由"记忆跨度" `H_i` 决定，`H_i` 在 log 刻度上遍历 `[2, 8192]`，
意味着**同一层内同时存在从"只记 2 个 token"到"记忆 8192 个 token"的全时间尺度**，
用于兼顾局部语法与跨段落、跨代码块的长距离依赖。

完整数学推导见 **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**。

---

## 原创组件

| 组件 | 方案 | 关键性质（可证） |
|---|---|---|
| 状态更新 | 多时间尺度凸组合滚动 | 状态有界，不发散；多尺度输出门控 |
| 非线性激活 | **SoftWedge** `f(z)=z+a·s(z)`，底式 `s(z)=½+½·z/(1+\|z\|)` | 导数 `f' ∈ (1, 1+a/2]`，**处处非零**，无梯度断崖 |
| 归一化 | **CenteredScaleNorm** | 分母设平滑下界 `s_min`，近似常量向量不被放大（深堆叠前提） |
| 初始化 | **谱锚定**（QR 正交切片，σ_max 直接锚定） | 条件数为 1，前向增益谱平坦；ρ=1.0 由实测标定 |
| 跨层传递 | 状态接力 + 特征残差流 | 状态尺度不随深度放大；残差直连兜住深度梯度（实测 −28% ppl） |
| 通道竞争 | 组内成对竞争门控 | 与 softmax 同语义，但 Jacobian 为 O(1) 而非 O(1/d) |
| 并行扫描 | 对数域闭式解 | 与逐步递推**代数等价**；块首状态可由块末**精确反解** |

**未复用**：ReLU / GELU / SiLU / Mish、LayerNorm / RMSNorm / InstanceNorm / GroupNorm、
Xavier / Glorot / He / MSRA 初始化、LSTM / GRU 门控公式、SSM / HiPPO 离散化。

---

## 附带实验

架构价值由实测数据说明，不接受宣称。全部实验共用同一 token 流、同一配置、同一 token 预算与线程数。

- **基线对照**：同计算栈参数量的标准 Transformer（用解析公式求解 `d_model`/`d_ff` 匹配参数量），比较每 token 训练耗时、验证困惑度、峰值内存。
- **同复杂度对比（核心）**：与同样 O(N) 的公开架构横向对比——**线性注意力 / RWKV / Mamba / Hyena / H3**。参数量用 meta device 实例化对齐（手推公式漏项严重），层数偏离不超 ±1，同语料同协议。手写实现，不调用第三方库。**实测结论：架构修正后 T=256 下升至七架构第二**（ppl 57.83，超越 Mamba/RWKV/Hyena/Transformer/线性注意力，仍不及 H3 53.93；§10、§12）。
- **消融实验**：逐个关闭 保护门 / 循环滤波 / 局部卷积 / 非线性支路，观察 loss 与速度变化。
- **种子稳定性**：多随机种子重复，检验可复现性。
- **缩放趋势**：small（1.4M）→ 20M 原型。
- **长序列验证（核心）**：固定模型、改变序列长度，实测前向耗时，验证 O(N) 优势与交叉点。**进程隔离后重测**：交叉点 T≈384（6 层对照）/ T≈600（4 层对照），T=4096 时每 token 比 Transformer 快 **22.7×**（早期未隔离的 12.6× 为下界）。
  > 该 22.7× 的条件必须写明：**batch=1、纯前向（不含反向）、8 线程、1.4M 小模型**，
  > 数据 `experiments/results/seq_scaling.json`，IQR 4%/3%。**不是训练速度**——
  > batch=8 训练步对 O(N) 同行仍落后（报告 §13）。融合改动后重测为 28.1×（§2.4 表）。
- **七架构速度缩放**：同复杂度五架构 + SunnyNet + Transformer，ms/token 随 T 同测。**前向**下 SunnyNet 仅在 512<T<2048 领先全部同行；T=4096 时 Hyena 仍快 1.9×、H3 追平。
- **训练速度专项（结果不利，如实记录）**：batch=8 **训练步（前向+反向+更新）**、每格独立子进程。**对 O(N) 同行全面落后 20–52%，从 T=256 到 T=4096 无交叉点**；只在 T=256 对 Transformer 领先 1.10×。根因是算子数量与块内宽度，**非计算量**（MAC/token 与 Mamba 相当）。
- **算子融合（本轮）**：`torch.compile` 不可用（MSVC 存在但 Windows SDK 头文件缺失），CPU 上唯一可行路径是 **oneDNN 现成核**——`CenteredScaleNorm` 改走 `F.layer_norm`（前向 19.5×），扫描改**两级并行**（bit-identical）。训练步同架构提速 **19–41%**，且质量同步提升（ppl 92.04→84.37）。见报告 §14.1。
- **减层加宽（本轮，原创）**：meta 计数发现**参数对齐下 SunnyNet 层数远超同行**（20M: L10 vs 5–7）。在栈参数 ±0.1% 前提下减层加宽（L10→L6、dim 512→640）：训练步 **−12.1%**、质量 **ppl −10.9%**、内存 **−14%**，层数回到同行档。见报告 §14.2–14.4。
- **规模化**：small → 20M → 100M（p100m，栈 93.7M）均已实测；**0.5B（栈 481.65M）本机不可实例化（OOM）**，趋势只到 100M。见报告 §14.5。
- **20M 同协议对照（旧"双输"结论推翻）**：栈参数对齐、T=256、200 步，SunnyNet 质量**升至五架构第一**（ppl 77.90，超越 rwkv/mamba/h3/hyena），速度劣势从 small 的 20–52% **收窄到 8–37%**。注：该变化为**减层加宽 + 规模**共同作用，两侧架构集合/步数不同，归因不可简化。见报告 §14.6。
- **长序列训练对照**：T=1024 真实训练下，SunnyNet 快 **2.08×**、ppl 优 **21.2%**、内存省 **36.8%**（此表为早期未隔离测量，绝对值以报告 §13 为准）。
- **长距离探针**：在真实验证语料上统计"重复上下文 top-1 命中率"，按距离分桶（near/mid/far/ultra）+ 多键长（1/4/8 token）。代码域远距命中 **0.13**（未训练模型为 0.0，探针有判别力）；自然语言域接近 0。

实测结果见 [docs/EXPERIMENT_REPORT.md](docs/EXPERIMENT_REPORT.md)。

**一句话结论**：经**算子融合 + 减层加宽**后，架构从"质量换速度"转为**"质量与速度双赢"**，但训练速度**仍未反超同行**。

- **前向**对 Transformer 在 T≳384 后更快（T=4096 快 22.7×）；对 O(N) 同行只在 **512<T<2048** 领先。
- **训练**（前向+反向）在同架构上已提速 19–41%（融合）+ 12–17%（减层加宽），
  但**对 Mamba/H3/Hyena 的 20–52% 差距尚未消除**——是**缩小**，不是反超。
- **质量**经修正后 T=256 升至七架构**第二**（ppl 57.83，仅落后 H3 53.93）；
  20M 减层加宽后 ppl **再降 10.9%**（54.82，400 步未收敛），并在 20M 同协议对照中**升至第一**。
- **scaling 趋势是变好而非变差**：small 质量第二 → **20M 质量第一**，速度劣势同步收窄（20–52% → 8–37%）。

**是否适用取决于任务同时要求"长序列 + 质量优先 + 可接受训练成本"。**

---

## 仓库结构

```
sunnynet/
  ops.py          原创算子: poly_sigmoid / softwedge / CenteredScaleNorm /
                  多尺度衰减 / 对数域并行扫描 / 通用门控扫描
  layers.py       SunnyBlock: 状态更新单元 + 循环滤波 + 局部卷积 + 非线性支路
  linear_baselines.py  O(N) 同复杂度对照: 线性注意力 / RWKV / Mamba / Hyena / H3
  model.py        SunnyNet 主体 (状态 (B,L,D), 逐层接力)
  init.py         谱锚定初始化
  baseline.py     同参数量 Transformer 对照 (仅用于对照, 非架构本体)
  tokenizer.py    从零实现的 BPE 分词器
  profiling.py    真实性能探针 (峰值内存 / CPU 缓存命中率 / 吞吐)
train/
  train.py        统一训练器 (两种架构共用同一循环)
  evaluate.py     困惑度 / 长距离探针 / 生成样例
experiments/
  run_experiments.py   实验编排 (baseline / ablation / seeds / scale / probe)
  battery.sh           批量运行基线 + 消融 + 种子 + 缩放
  battery2.sh          20M 缩放对照 + 长序列训练对照
  long_sequence.py     前向延迟 vs 序列长度 (O(N) 验证) + 长序列训练
  make_figures.py      生成全部图表 -> docs/figures/
  reduce_results.py    分散结果聚合为 report_*.json (供出图)
  collect_results.py   汇总所有结果为 Markdown 表
tests/
  test_architecture.py 架构自检 (分块一致性 / 因果性 / 梯度界 / 深度稳定, 41 项)
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

SunnyNet 为独立原创架构。需要如实指出的是：

- 其状态递推**在数学形式上与线性注意力类工作有亲缘性**（都是一阶线性递推），
  这一点不回避；
- 但门控构造（凸组合 + 对数均匀多尺度 + 逐元素保护门）、层间传递（状态接力 +
  特征残差流）、激活函数、归一化、初始化均为自行设计与实测标定；
- 未参考或移植任何既有项目的代码与结构。

架构设计过程中有**四处结论被实测推翻并修正**（softmax 通道竞争导致梯度随深度归零；
谱半径层深衰减的符号设反；CenteredScaleNorm 的 $s_{\min}$ 高于实际激活尺度导致归一化
退化为常数缩放；原设计排除残差反而造成最大单项质量损失），修正依据与数据记录在
docs/ARCHITECTURE.md 第 3/4/5 节。保留这些记录是为了说明各设计约束的来源，
而不是事后美化。

---

## 风险与局限

1. **短序列劣势（重要）**：O(N) 优势**只在长序列兑现**。进程隔离重测（batch=1 前向）交叉点
   **T≈384（6 层对照）/ T≈600（4 层对照）**：短于此，SunnyNet 慢于标准 Transformer；
   长于此，SunnyNet 更快（T=4096 时每 token 快 22.7×）。
   **该架构不适合短序列场景。**
   > 20M 规模下的"质量与速度双输"是**修正前栈 + 层数未重平衡**时的旧结论，
   > 已被 §14.6 推翻（重平衡后质量升至五架构第一）。此处保留原记录以便追溯。
2. **短序列下质量仍不及 H3（实测，已大幅改善）**：T=256、600 步的同协议对比中，
   架构修正（残差 + CSN 地板，见 §10/§12）后 SunnyNet ppl 由 93.82 降至 **57.83**，
   升至七架构**第二**，超越 Mamba(63.34) / RWKV(64.10) / Hyena(82.50) /
   Transformer(90.05) / 线性注意力(91.91)，但仍不及 H3（53.93，差约 7%）。
   该对比**未收敛、单种子、学习率未逐架构调优**，
   只能读作早期样本效率排序（详见 docs/EXPERIMENT_REPORT.md §10、§12）。
3. **长序列对同行也无绝对优势（实测）**：T=4096 时 Hyena 仍快 1.9×、H3 基本追平；
   对全部同行领先的区间只有 **512<T<2048**。
4. **早期耗时数据失真（已修正）**：未做进程隔离时同进程累积构建会使读数偏高达 2 倍以上，
   且放大倍数随架构而变。§2.1 的绝对值与 12.6× 应视为下界，见报告 §2.4。
5. **长距离依赖**：滚动状态的信息容量受限于固定维度，理论上界低于全注意力
   （`O(T)` 可寻址）。大跨度通道只保证"不被遗忘"，不保证"被检索"。
6. **伪收敛**：loss / ppl 下降不等于具备推理与指令跟随能力。
7. **语料覆盖**：中文维基百科在本机网络不可达，因此**缺少"百科"域**，
   实验只覆盖 古籍 / 英文文学 / 代码 三个真实域。
8. **规模**：仅完成 1.4M（1200 步）与 20M（300 步，未收敛）的初步验证。
   放大到 0.5B 的结论**未做任何实验**。
9. **消融未成结论**：600 步未收敛，关组件后 ppl 反而更低，不构成组件有效性证据。
10. **同复杂度对比未收敛**：七架构对比仅 600 步，都未收敛，且未逐架构调优学习率，
    架构间小差距落在种子噪声范围内。
11. **`torch.compile` 不可用**：本机无 MSVC，算子融合这条优化路线无法验证。

详见 **[docs/EXPERIMENT_REPORT.md](docs/EXPERIMENT_REPORT.md)** 与 **[safety_notice.md](safety_notice.md)**。
模型为实验科研项目，**中医相关输出不可作为医疗诊断依据**。

---

## 赞助

如果这个项目对你有帮助，欢迎支持后续研发：

<a href="https://afdian.com/a/sunny-cx">
  <img src="https://img.shields.io/badge/爱发电-赞助支持-946CE6?style=for-the-badge" alt="爱发电">
</a>

---

## 许可证

[Apache License 2.0](LICENSE)

Copyright 2026 sunny-cx