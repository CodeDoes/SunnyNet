# 社区发布材料（草稿）

> **使用说明**：以下内容由项目作者本人以真实身份发布。所有数字均来自仓库
> `experiments/results/` 下的原始 JSON，可逐条复核。发帖时请署本人 ID
> `SunnyClaw-cx`，并如实说明这是自荐项目（多数社区要求 disclose）。
> **不要**用机器人账号发布、不要在多个仓库重复刷同样内容、不要删掉"局限"部分。

---

## 一、英文版（适合 GitHub Discussions / r/MachineLearning / HN Show HN）

### 标题备选

- `SunnyNet: an O(N) CPU-first sequence architecture — 22.7× faster than a size-matched Transformer at T=4096 (forward), but slower than Mamba/RWKV at short T. Looking for feedback.`
- `An O(N) linear-recurrence architecture with pairwise competition gating — measured scaling data, including results where it loses`

### 正文

Hi all — I've been building an original O(N) sequence architecture aimed at CPU
training, and I'd like to share the *measured* results, including the parts that
don't flatter it. Everything below is reproducible from the repo's
`experiments/results/*.json`.

**Repo**: https://github.com/SunnyClaw-cx/SunnyNet (Apache-2.0)

#### What it is

A linear-recurrence architecture, not a Transformer/RNN/Mamba/CNN derivative.
The state update is a first-order linear recurrence
`s_t = p_t · s_{t-1} + (1 - p_t) · u_t`, where the forget factors `p` come from
log-uniform multi-scale horizons and a *pairwise intra-group competition gate*
(`sigmoid(g_i-g_j)·sigmoid(g_i-g_k)`), rather than softmax. Blocks also carry a
depthwise causal conv, a 3-tap circulant state filter, and a residual stream.
I'm aware the recurrence form is related to linear-attention work — I say so in
the README rather than hiding it.

#### Where it wins (measured)

Forward latency, batch=1, 8 threads, size-matched (~365k param compute stack),
isolated subprocess per cell, median of 9–15 runs
(`experiments/results/seq_scaling.json`):

| T | SunnyNet | Transformer | ratio | IQR |
|---|---|---|---|---|
| 512 | 26.4 ms | 40.8 ms | 1.55× | 12.4%/6.0% |
| 1024 | 50.0 ms | 141.2 ms | 2.83× | 6.3%/6.6% |
| 2048 | 98.1 ms | 638.2 ms | 6.51× | 6.7%/6.3% |
| 4096 | **148.7 ms** | **3373.4 ms** | **22.69×** | 3.9%/2.7% |

At 20M params, size-matched against hand-written Mamba/H3/Hyena/RWKV baselines
(same corpus, 200 steps, T=256), it came out ahead on validation ppl:

| arch | stack params | ppl |
|---|---|---|
| SunnyNet | 16,755,230 | **77.90** |
| RWKV | 16,844,768 | 79.12 |
| Mamba | 16,763,712 | 81.92 |
| H3 | 16,862,256 | 86.99 |
| Hyena | 16,801,776 | 109.07 |

#### Where it loses (measured — please read)

- **Training step (fwd+bwd) is still slower than Mamba/H3/Hyena**: 8–37% at 20M,
  20–52% at 1.4M. There is **no crossover** on training speed up to T=4096.
  The 22.7× is *forward-only latency*, not training throughput.
- **Short sequences are bad**: crossover vs Transformer is around T≈384
  (6-layer control). Below that it's slower.
- **Quality gap at small scale**: at 1.4M/T=256 it's 2nd of 7 archs, ~7% behind
  H3. The comparison is not converged and mostly single-seed.
- **0.5B doesn't run here**: 15.9 GB machine, the 481M-param stack OOMs on
  instantiation. So the scaling story only goes up to 100M.

#### Two things I found that might interest others

1. **Layer count under parameter matching.** When you size-match by compute-stack
   params, low-rank narrow designs end up needing *far more layers*
   (10 vs 5–7 at 20M). That's a real efficiency tax. Re-balancing to
   `dim 512→640, L 10→6` at equal params gave **12% faster steps, 10.9% better
   ppl, 14% less memory** — a free lunch I'd missed for a while.
2. **Operator fusion ceiling on CPU.** I measured how much a fully fused
   scan/competition kernel could possibly buy by replacing both with trivial ops:
   **8.8% end-to-end ceiling** on the training step. Since the backward pass is
   ~65% of the step, fusing the forward scan alone can't move the needle much.
   (torch.compile isn't available here — MSVC present but Windows SDK headers
   missing.)

#### Questions for this community

- The pairwise-competition gate is my attempt to avoid softmax's `1/d` Jacobian
  decay in deep stacks. Is there prior art I should cite? I want to credit
  correctly.
- For Mamba/RWKV authors: my training-step gap comes mostly from operator count
  and narrow inner dim, not FLOPs (MAC/token ≈ 8064 vs Mamba's 7029). Any
  suggestions on where a hand-written CPU implementation usually loses most?
- Anyone seen the "more layers under param matching" effect quantified for
  low-rank linear architectures? I'd like to compare notes.

Happy to run additional controls if anyone points out a fairness problem in the
setup — I'd rather fix the comparison than defend the number.

---

## 二、中文版（适合知乎 / V2EX / 技术群）

### 标题备选

- `我做了一个原创 O(N) 序列架构，CPU 上 T=4096 前向比 Transformer 快 22.7 倍——但训练比 Mamba 慢，数据都放出来了`
- `SunnyNet：一份"包含失败结果"的 O(N) 架构实测报告`

### 正文

先说结论，再说条件，最后说它不行的地方。

**项目**：https://github.com/SunnyClaw-cx/SunnyNet （Apache-2.0，自行实现，未移植任何现有项目代码）

**架构**：一阶线性递推 + 对数均匀多尺度遗忘 + **组内成对竞争门控**
（`sigmoid(g_i-g_j)·sigmoid(g_i-g_k)`，替代 softmax：softmax 的 Jacobian 含 1/d，
深层堆叠会让梯度归零，这是我实测撞到的）。另有深度可分因果卷积、三抽头循环状态滤波、
特征残差流。递推形式与线性注意力类工作有亲缘性，这一点我在 README 里直接写明，不回避。

**赢的部分（实测）**

1.4M 参数、batch=1 纯前向、8 线程、每格独立子进程、9–15 次取中位
（原始数据 `experiments/results/seq_scaling.json`）：

| T | SunnyNet | Transformer | 倍数 | IQR |
|---|---|---|---|---|
| 512 | 26.4 ms | 40.8 ms | 1.55× | 12.4%/6.0% |
| 1024 | 50.0 ms | 141.2 ms | 2.83× | 6.3%/6.6% |
| 2048 | 98.1 ms | 638.2 ms | 6.51× | 6.7%/6.3% |
| 4096 | **148.7 ms** | **3373.4 ms** | **22.69×** | 3.9%/2.7% |

20M 参数、栈参数对齐、同语料 200 步、T=256，ppl：

| 架构 | 栈参数 | ppl |
|---|---|---|
| SunnyNet | 16,755,230 | **77.90** |
| RWKV | 16,844,768 | 79.12 |
| Mamba | 16,763,712 | 81.92 |
| H3 | 16,862,256 | 86.99 |
| Hyena | 16,801,776 | 109.07 |

**输的部分（如实说）**

- **训练步（前向+反向）仍比 Mamba/H3/Hyena 慢**：20M 规模慢 8–37%，1.4M 规模慢 20–52%，
  到 T=4096 都没有交叉点。**22.7× 是前向延迟比，不是训练速度比。**
- **短序列不行**：与 Transformer 的交叉点约 T≈384。短于它更慢。
- **小规模质量只排第二**：1.4M/T=256 七架构对比，落后 H3 约 7%，且未收敛、以单种子为主。
- **0.5B 跑不了**：15.9GB 机器，481M 栈参数实例化直接 OOM，scaling 只能讲到 100M。

**两个我觉得别人可能用得上的发现**

1. **参数对齐下层数会失控**：按计算栈参数对齐时，低秩窄模型被迫堆很多层
   （20M 下 10 层 vs 同行的 5–7 层）。我把它重平衡到 `dim 512→640, L 10→6`，
   同等参数量下**训练快 12%、ppl 好 10.9%、内存省 14%**。
2. **CPU 上算子融合的收益上界**：我把扫描和竞争门控都替换成极简算子，量出
   "完全融合"的端到端上限只有 **8.8%**（反向占训练步 ~65%，前向占比再高也摊不进大分母）。
   本机 `torch.compile` 不可用（MSVC 在，但 Windows SDK 头文件缺失）。

**想请教的问题**

- 成对竞争门控这个构造，有没有我该引用的先行工作？我想把引用弄对。
- 手写 CPU 实现里，低秩线性架构一般在哪里丢分最多？我的 MAC/token 约 8064（Mamba 7029），
  差距应该在算子数和内层宽度，不在算力。

**如果谁发现我的对照协议有不公平的地方，请指出，我更愿意修对照而不是辩护数字。**

---

## 三、发帖前自检清单

- [ ] 所有数字与 `experiments/results/*.json` 对得上（**不要凭记忆写数字**）
- [ ] 保留"输的部分"整段，不得删除
- [ ] 明示"这是作者自荐"，遵守各社区 self-promotion 规则
- [ ] 同一内容不在多个仓库重复刷；在 Mamba/RWKV 仓库发帖时**以提问为主**，
      介绍自己的项目为辅
- [ ] 不用机器人账号、不批量刷帖、不买赞
- [ ] 英文帖不要机翻痕迹过重（可请母语者过一眼）