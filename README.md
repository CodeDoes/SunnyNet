# SunnyNet

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

```bash
export SUNNY_W64=F:/devtools/w64devkit
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
  test_architecture.py  架构自检 (分块一致性 / 因果性 / 梯度界 / 深度稳定, 59 项)
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