# SunnyNet 发帖包（链接已验证 + 成稿可复制）

> 验证日期 **2026-10-21**。每一行链接都是**实际打开过**的，不是推测。
> 所有数字取自 `docs/EXPERIMENT_REPORT.md` §16 与 `experiments/results/*.json`。

---

## 0. 更正：上一版有一处写错了

**上一版说「RWKV-LM 没有 Discussions」——这是错的。**

实测：`https://github.com/BlinkDL/RWKV-LM/discussions` 正常存在，
分类有 **Announcements / General / Ideas / Q&A / Show and tell**，
且 `Show and tell` 里 `#189` 就是第三方实现帖（jackd 的「Numerically stable
parallel cumsum-based WKV + jax/tf/keras implementations」）。

不过 RWKV 作者在 `#174` 欢迎帖里明确写了：
> PS: its generally better to ask such questions on our discord - https://discord.gg/bDSBUMeFpc

⇒ **两处都能发，但 Discord 是主渠道**，Discussions 次之。

---

## 1. 链接总表（全部已验证）

| # | 社区 / 入口 | 链接 | 能不能发自己项目 | 备注 |
|---|---|---|---|---|
| 1 | mamba · Show and tell | https://github.com/state-spaces/mamba/discussions/categories/show-and-tell | **能** | 同区已有 `#865` Flax 实现帖、`#844` Routed Attention 帖 |
| 2 | mamba · Ideas | https://github.com/state-spaces/mamba/discussions/categories/ideas | **能** | `#1051`（2026-10-01）第三方自述 S-Mamba 求反馈 |
| 3 | **RWKV-LM · Show and tell** | https://github.com/BlinkDL/RWKV-LM/discussions/categories/show-and-tell | **能** | 见上方更正；`#189` 有先例 |
| 4 | RWKV 官方 Discord | https://discord.gg/bDSBUMeFpc | **能** | 作者主推渠道；**Discord 是聊天不是论坛，不能贴长文 → 见 `COMMUNITY_DISCORD_GUIDE.md`** |
| 5 | **Awesome-LA-Papers** | https://github.com/HallerPatrick/Awesome-LA-Papers | **能（PR）** | README 有独立 `## Contributing` 节，明文 "We welcome contributions!"，且含 `Libraries and Implementations` 栏目 |
| 6 | aisys-building-blocks | https://github.com/YiwenC25/aisys-building-blocks | **能（PR）** | "If you think we're missing something, please open an issue or PR!" |
| 7 | HazyResearch/zoology | https://github.com/HazyResearch/zoology | **能（限合成任务）** | "We wholeheartedly welcome contributions for new synthetics (MQAR variants)" — 需先跑 MQAR |
| 8 | fla-org/flash-linear-attention | https://github.com/fla-org/flash-linear-attention/discussions | **谨慎** | Show and tell 分类存在；但该库收的是 **Triton 实现**，直接贴仓库链接易被忽略 → **先问再发** |
| 9 | Show HN | https://news.ycombinator.com/submit | **能** | ⚠️ 见 §6 警告 |
| 10 | r/MachineLearning 自荐帖 | https://www.reddit.com/r/MachineLearning/ | **每周置顶帖内能** | 主 feed 禁直接自荐 |

**明确不要碰的**：
- RWKV README 的 "400+ RWKV projects" —— 那是 GitHub 关键字搜索结果页，**不是投稿列表**
- `radarFudan/Awesome-state-space-models` 直接提 PR —— README **没有** Contributing 节，且是**论文**列表；SunnyNet 无 paper ⇒ **先开 issue 问**，别直接 PR

---

## 2. mamba Discussions · Show and tell（成品，可复制）

**标题**
```
An O(N) CPU-first linear recurrence — measured step time now ties RWKV at T=256, plus the numbers where it loses
```

**正文**
```
Hi — I've been writing a from-scratch O(N) sequence architecture aimed at CPU
training, and I hand-wrote size-matched baselines for Mamba / RWKV / H3 / Hyena /
linear attention to compare against. Sharing the measurements, including the ones
that don't flatter my side.

Repo (Apache-2.0): https://github.com/SunnyClaw-cx/SunnyNet

The recurrence is first-order: s_t = p_t · s_{t-1} + (1 - p_t) · u_t, with p coming
from log-uniform multi-scale horizons and an intra-group pairwise competition gate
(sigmoid(g_i-g_j)·sigmoid(g_i-g_k)) instead of softmax. I'm aware the recurrence form
is related to linear-attention work — it's stated plainly in the README rather than
glossed over.

Training step, fwd+bwd+update, size-matched ~365k compute-stack params, isolated
subprocess per cell, 5 rounds, median (ms/token, lower is better):

  arch                  T=256     T=512
  SunnyNet (dim=80)     0.0953    0.0986
  RWKV                  0.0982    0.0951
  H3                    0.1138    0.1239
  SunnyNet (dim=128)    0.1158    0.1243
  Mamba                 0.1292    0.1271
  Hyena                 0.1312    0.1468
  linear attention      0.1569    0.1676

Two equivalent changes got it from "20-52% behind everyone" to here:
  1. An unreachable clamp. p = 1 - delta*(1-sigmoid(gate)) with delta <= 1/h_min, so
     p >= 1 - 1/h_min = 0.5, while the floor is 1/h_max = 4.88e-4. I sampled the real
     range as [0.5902, 0.99994] — the bound is 1209x below the actual minimum. Removing
     it changes the forward by exactly 0.0 and drops a ClampBackward node.
  2. Narrower residual stream at equal stack params. The per-element ops pay by
     activation width, so dim 128 -> 80 buys ~18% and costs quality.

Where it loses (please read this part too):

- "Ties RWKV" is statistical parity, not a win. Over 9 paired rounds it won 3/9 at
  T=256 and 4/9 at T=512; the paired-ratio median is 1.042 / 1.006, while the
  ratio-of-medians favors it. Three metrics disagree, so I write "ties".
- Parity only holds at L=4. At L=6 (matching RWKV's depth) every equal-budget narrow
  config is 30-40% slower. My alignment protocol matches stack params, not layer
  count — so the tie relies on using fewer layers. I think this is the fair objection.
- Quality costs. The narrow config's 600-step ppl is 62.37 vs 54.89 for dim=128.
  Speed and quality moved in opposite directions.
- Short sequences are bad: crossover vs a size-matched Transformer is around T≈384.

Question I'd genuinely like help with: profiling shows aten::mm costs roughly the
same across all these architectures, but my per-element ops (mul, cat) are 3-6x
RWKV's. My MAC/token is ~8064 vs Mamba's ~7029, so the gap isn't FLOPs. For people
who've written CPU-side implementations — where do hand-written versions usually
lose the most? I'd rather fix the implementation than defend the number.

Happy to run extra controls if anyone spots a fairness problem in the setup.
```

---

## 3. RWKV-LM Discussions · Show and tell（成品，可复制）

**标题**
```
RWKV as a step-time baseline: an O(N) recurrence that ties it at T=256, and where it doesn't
```

**正文**
```
Hi — I used RWKV as one of my main baselines for a CPU-first O(N) architecture I've been
building, and I wanted to share the head-to-head plus a specific question about the CPU
side of the recurrence.

Repo (Apache-2.0): https://github.com/SunnyClaw-cx/SunnyNet

Step time, fwd+bwd+update, ~365k compute-stack params, isolated subprocess, 5-round median:

  T=256:  my narrow config 0.0953 vs RWKV 0.0982 ms/token  (I'm nominally ahead)
  T=512:  my narrow config 0.0986 vs RWKV 0.0951 ms/token  (RWKV ahead)

Over 9 paired rounds the sign test went 3/9 (T=256) and 4/9 (T=512), so I report this
as a tie rather than a win. At T=512 RWKV is ahead.

The specific thing I wanted to ask: I found that exp(-P) is 2.2x slower than exp(P) in
my scan, and replacing it with reciprocal(exp(P)) is numerically fine (fp64 relative
error 7.6e-17, fp32 6.6e-08) — but end to end it only helps at T=1024, not at T=256/512.
Is that consistent with what you see on the RWKV side? I suspect it's a small-tensor
scheduling effect dominating below a certain T, but I don't have a measurement that
separates the two.

One more data point that might interest the community: when I size-matched by compute-stack
params, my low-rank design needed far more layers (10 vs 5-7 at 20M scale). Re-balancing
to dim 512->640, L 10->6 at equal params gave 12% faster steps and 10.9% better ppl.
I'd be curious whether RWKV's depth was tuned the same way.

(Usually pointed at Discord — happy to move this there if that's preferred.)
```

---

## 4. Awesome-LA-Papers · PR（成品）

这是**最值得做的一个** —— 该列表有独立的 `## Contributing` 节，明文欢迎提交。

**改动**：在 `## Libraries and Implementations` 节、`InLine` 条目之后加一行：

```markdown
- **SunnyNet**: An original O(N) linear-recurrence architecture targeting CPU-first training, with hand-written size-matched baselines (Mamba / RWKV / H3 / Hyena / linear attention) and a reproducibility suite.  
  [GitHub Repository](https://github.com/SunnyClaw-cx/SunnyNet)
```

（注意：原文件该栏每条**末尾有两个空格**做换行，照抄格式。）

**PR 标题**
```
Add SunnyNet to Libraries and Implementations
```

**PR 描述**
```
Adds SunnyNet, an original O(N) linear-recurrence architecture designed for CPU-first
training, under Libraries and Implementations.

It fits the Libraries section because it ships a full baseline suite rather than a model
alone: hand-written size-matched implementations of Mamba, RWKV, H3, Hyena and linear
attention (dimension-matched via meta-device parameter counting, not hand-derived
formulas), plus the isolated-subprocess benchmark harness and the architecture invariant
tests used in its report.

Note: the recurrence form is a first-order linear recurrence, which I state explicitly in
the README as being related to linear-attention work rather than claiming independence
from it.

License: Apache-2.0.
```

---

## 5. aisys-building-blocks · PR（成品）

**改动**：在 `Can We Replace Attention?` → Papers 段末尾加：

```markdown
- SunnyNet — an O(N) linear-recurrence architecture written from scratch for CPU-first
  training, with measured comparisons against Mamba / RWKV / H3 / Hyena sized-matched at
  equal compute-stack params. [Repo](https://github.com/SunnyClaw-cx/SunnyNet)
```

**PR 标题**
```
Add SunnyNet to "Can We Replace Attention?"
```

**描述**
```
SunnyNet is an O(N) linear-recurrence architecture, developed independently and targeted
at CPU-first training (no GPU/CUDA dependency). It's listed because it comes with
size-matched hand-written baselines rather than only a model.

Honest framing, since this list is curated: it is not SOTA. At T=4096 its forward latency
is 22.7x a size-matched Transformer, but its training step only reaches statistical
parity with RWKV at T=256 and that parity holds only at L=4 — at matching depth it is
30-40% slower. The crossover against a Transformer is around T≈384, so it is not a
short-sequence win.

Apache-2.0. [Repo](https://github.com/SunnyClaw-cx/SunnyNet)
```

> 这个列表是"精选"性质，所以**主动写不利结果**反而是被收录的关键——
> 它证明你在做研究而不是打广告。

---

## 6. Show HN ⚠️ 有两条硬规则，必须先知道

**规则一（来自官方 Show HN Guidelines，经多方引述核对）**
- "Show HN is for something you've made that other people can play with."
- 必须是**你自己的**、**非琐碎的**、**已可试用的**、**尽量无门槛试用**
- **标题必须以 `Show HN` 开头**
- 博客文章 / 落地页 / 订阅页 / **列表** 属于 off topic —— 不能发 Show HN
- 不许求 upvote / 求评论；不许用公司或项目名当用户名

SunnyNet 是开源仓库，**"opensource repo people can run" 明确符合**，可以发。

**规则二（这条我必须直说）**

HN 官方 tips 明确写着：**不要用 LLM 撰写、甚至不要用 LLM 编辑正文**；
guidelines 也**禁止 AI 生成或 AI 编辑的帖子**。

⇒ **这一份我不能再替你写成稿。** 上面 §2/§3 的稿子你可以改用到别处，
但 **HN 那份正文需要你自己手写**。这不只是合规问题：HN 读者对机器腔的识别很快，
被识破的帖子会被 flag 掉，反而比不发更差。

我能提供的是**结构和素材**，你自己组织语言：

**标题**（格式必须是 `Show HN: 名称 – 一句话做什么`）
```
Show HN: SunnyNet – an O(N) sequence architecture that trains on CPU, no GPU
```

**正文要点（按这个顺序自己写）**
1. 一句话说它是什么
2. **为什么做**（真实的起因，不是 TAM）
3. 技术要点：一阶递推 + 成对竞争门控替代 softmax；为什么不用 GPU
4. **已知不行的部分先说**：与 Transformer 交叉点 T≈384，短序列更慢；
   与 RWKV 只是统计持平且仅 L=4 成立；质量比 dim=128 配置差
5. **一个具体的问题**（比 "thoughts?" 有效得多）

**要点**：CPU-only 是 HN 的钩子；主动认输是可信度来源；
提交后**发一条 maker comment 并守满 4–6 小时**；周二至周四美东 8–11 点发。

---

## 7. r/MachineLearning（成品，用于每周自荐帖）

主 feed 禁直接自荐。先找到当周 `[D] Self-Promotion Thread` 置顶帖再发。
若要发主 feed，必须带 `[P]` flair 且满足"开源 + 基准 + 局限"三件套。

```
[P] SunnyNet: an O(N) linear-recurrence architecture for CPU-first training

Repo: https://github.com/SunnyClaw-cx/SunnyNet (Apache-2.0)

Approach: first-order linear recurrence s_t = p_t·s_{t-1} + (1-p_t)·u_t, where p comes
from log-uniform multi-scale horizons and an intra-group pairwise competition gate
(sigmoid(g_i-g_j)·sigmoid(g_i-g_k)) rather than softmax. Plus a depthwise causal conv,
a 3-tap circulant state filter, and a residual stream. The recurrence form's relationship
to linear-attention work is stated in the README, not glossed over.

Results (size-matched ~365k compute-stack params, isolated subprocess per cell, median):

  step time ms/token    T=256    T=512
    SunnyNet dim=80     0.0953   0.0986
    RWKV                0.0982   0.0951
    H3                  0.1138   0.1239
    Mamba               0.1292   0.1271

  forward latency, batch=1:  T=512 1.55x, T=1024 2.83x, T=2048 6.51x, T=4096 22.69x
  vs a size-matched Transformer

Limitations:
  - "ties RWKV" is statistical parity only: 3/9 and 4/9 paired rounds won, ratio-of-medians
    and paired-ratio medians disagree
  - parity holds only at L=4; at matching depth (L=6) it's 30-40% slower
  - quality trades down vs the wider config: 600-step ppl 62.37 vs 54.89
  - crossover vs Transformer is T≈384, so short sequences are a loss
  - single implementation, lr not per-architecture tuned, peers mostly single-seed

Happy to be told the protocol is unfair somewhere — I'd rather fix the comparison.
```

---

## 8. 发帖顺序与硬约束

| 时间 | 动作 |
|---|---|
| D0 | 提 **Awesome-LA-Papers** PR（最高成功率） |
| D0 | 提 **aisys-building-blocks** PR |
| D+1 | mamba `Show and tell` 发帖 |
| D+2 | RWKV `Show and tell` 发帖（或 Discord 先问） |
| D+4 | 开 **issue** 问 `radarFudan/Awesome-state-space-models` 是否收无 paper 的仓库条目 |
| D+5 | fla Discord 先问，视回应再决定 |
| D+7 | **你自己手写** Show HN（周二至周四美东上午） |
| D+10 | r/MachineLearning 每周自荐帖 |

**硬约束（会被判 spam 的行为）**
- 同一份正文在两个社区同时出现（Reddit 明确把同日跨版同文判为 spam）
- 同一天在 ≥3 个社区发同类内容
- 新账号首发即推广（建议账号先有 10+ 条真实技术评论）
- 求赞、互赞、批量发

**每帖必做**
- 署真实 ID `SunnyClaw-cx`，明示这是自己的项目
- 保留"输的部分"整段
- 发出后守评论区 2 小时以上
- 结尾是一个**真问题**，不是"求 star"