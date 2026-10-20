# SunnyNet 社区推广路线（真实入口 + 分渠道稿件）

> 本文只记录**已实际打开验证过**的入口，标注验证日期 2026-10-21。
> 未经打开的渠道一律标 `[未验证]`，不得当成事实使用。

---

## 0. 一条不能越的线

「多账号灌水 + 不让人看出是谁发的」**不做**。理由是它会直接损害项目本身：

| 做法 | 实际后果 |
|---|---|
| 同文多仓库重复贴 | GitHub 判 spam，**连带作者账号**受限 |
| 新账号首发即推广 | 触发 spam 过滤（strict 版主会自动拦截零评论历史的账号） |
| 隐瞒作者身份 | 多数社区**明确要求 disclose**，隐瞒一旦被点破，技术可信度归零 |
| 买赞/互赞 | HN 与 Reddit 都有专门的反作弊检测 |

而 SunnyNet 手里是**可复核的实测数据**——藏起来发等于把最强的资产丢掉。
下面所有稿件都**署真实 ID `SunnyClaw-cx`** 并**明示自荐**。

**「不像是刷的」的正确做法**不是伪装身份，而是写成人话：

- 第一人称，讲自己撞到什么、为什么改
- 数字带条件（batch、T、线程、几轮），能被追问
- **主动写输的部分**——技术社区靠这个判断你是不是真的做过
- 结尾提一个**真实的技术问题**，而不是「求 star」
- 不用「revolutionary / 颠覆 / 全面领先」这类词，一秒暴露

---

## 1. 真实入口清单（已验证）

### A 类：直接欢迎贴自己项目的仓库

| 社区 | 入口 | 自荐许可 | 依据 |
|---|---|---|---|
| **state-spaces/mamba** → `Show and tell` | https://github.com/state-spaces/mamba/discussions/categories/show-and-tell | **可以** | 分类页存在且为开放讨论格式；同区已有第三方项目帖 |
| **state-spaces/mamba** → `Ideas` | https://github.com/state-spaces/mamba/discussions/categories/ideas | **可以** | 见实例 #1051（2026-10-01，作者自述 S-Mamba 原型并求反馈） |
| **HallerPatrick/Awesome-LA-Papers** | https://github.com/HallerPatrick/awesome-la-papers | **可以** | README 尾明确 "We welcome contributions!"，含 Libraries and Implementations 栏目 |
| **YiwenC25/aisys-building-blocks** | https://github.com/YiwenC25/aisys-building-blocks | **可以** | README "If you think we're missing something, please open an issue or PR!" |
| **HazyResearch/zoology** | https://github.com/HazyResearch/zoology | **可以（限合成任务）** | README Notes：*"We wholeheartedly welcome contributions for new synthetics (MQAR variants) — feel free to make a PR or email us!"* |
| **radarFudan / dongzhuoyao - Awesome-state-space-models** | https://github.com/radarFudan/Awesome-state-space-models | **可以（PR 补条目）** | 论文/仓库精选表，含 unofficial 实现栏 |

### B 类：没有 GitHub Discussions，走外部社区

| 社区 | 入口 | 说明 |
|---|---|---|
| **RWKV 官方 Discord** | https://discord.gg/bDSBUMeFpc | RWKV-LM README 上的官方邀请链接。**已确认 RWKV-LM 没有 Discussions 标签页**，全部讨论在 Discord |
| **RWKV 官网 / Wiki** | https://rwkv.com · https://wiki.rwkv.com | wiki 有 community / code-of-conduct 页 |
| **HazyResearch 系（H3/Hyena）** | https://github.com/HazyResearch/safari | 仓库 README 明确 *"We're happy to share independent reimplementations and explainer posts"* |

**⚠️ 关键澄清**：RWKV 仓库里那份 "400+ RWKV projects" 列表是**用 GitHub 搜索 `rwkv` 关键字生成的**，不是投稿列表。SunnyNet **不是 RWKV 项目**，硬塞进去就是错误归类 —— 不要做。

### C 类：通用技术社区（需遵守自荐比例规则）

| 社区 | 入口 | 规则要点 |
|---|---|---|
| **Hacker News — Show HN** | https://news.ycombinator.com/showhn.html | 必须是**自己的作品**且**能直接试用**。SunnyNet 的 CPU-first 属性在这里是强卖点（HN 对「不用 GPU」极敏感） |
| **r/MachineLearning** | https://reddit.com/r/MachineLearning | **主 feed 禁直接自荐**。用每周 Self-Promotion 置顶帖；开源项目发布可用 `[P]` flair。建议账号先有 10+ 条真实技术评论再发 |
| **r/LocalLLaMA** | https://reddit.com/r/LocalLLaMA | 对「本地/自己硬件上跑」的内容宽容度更高，适合 CPU 训练这个点 |
| **知乎** | 写文章而非发想法 | 算法惩罚纯推广；`包含失败结果的实测报告` 这类标题点击率反而高 |
| **V2EX → 分享创造** | https://v2ex.com/go/create | 中文技术圈对「全面领先」高度警惕，但吃「有数据 + 认输」这一套 |

### D 类：高杠杆但慢（建议列入下一阶段，不是本轮）

| 渠道 | 为什么值得 |
|---|---|
| **arXiv 技术报告 + Papers With Code** | 唯一能让引用沉淀下来的形式。当前不满足「同行评审」，但可作为 preprint 技术报告 |
| **HuggingFace Hub 放小 checkpoint** | 可被搜索、可被 `from_pretrained` 加载，是长期自然流量入口 |
| **fla-org/flash-linear-attention** | 该库是同类架构的实现集合，收录门槛高（要有 Triton kernel）——**现在不满足**，等有 kernel 再谈 |

---

## 2. 分渠道稿件

> 通用正文见 `docs/COMMUNITY_POST_DRAFT.md`（**已同步 §16 最新数据**）。
> 每到一个社区必须**改写开头与结尾**，不能原文照搬 —— 这既是礼貌，也是避免被判 spam 的技术要求。

### 2.1 mamba Discussions · Show and tell

标题：
```
An O(N) CPU-first linear-recurrence architecture — measured step-time now matches RWKV, and I'm posting the numbers where it loses too
```

正文要点顺序（**不要**以"看我的项目"开场）：

1. 先说**读了什么**：mamba 的 selective scan 让状态参数依赖输入，这是我做对照实现的起点
2. 再说**怎么做对照的**：手写 Mamba/RWKV/H3/Hyena 基线，栈参数对齐到 ~365k，每格独立子进程
3. 放**双方数字**（下表）
4. 写**我输在哪**：短序列慢、L=6 时慢 30–40%
5. 提问：`aten::mm` 各架构耗时几乎相同，但我的 `mul`/`cat` 是 rwkv 的 3–6 倍 —— 手写实现常在哪丢分？

| 架构 | 栈参数 | T=256 ms/tok | T=512 ms/tok | 600 步 ppl |
|---|---|---|---|---|
| SunnyNet `smalln80` | 365,940 | **0.0953** | 0.0986 | 62.37 |
| RWKV | 361,920 | 0.0982 | **0.0951** | 63.86 |
| H3 | 369,760 | 0.1138 | 0.1239 | **53.93** |
| Mamba | 364,960 | 0.1292 | 0.1271 | 63.34 |
| Hyena | 392,760 | 0.1312 | 0.1468 | — |
| Linear attn | 364,720 | 0.1569 | 0.1676 | 91.91 |

### 2.2 RWKV Discord

**注意**：Discord 上先聊天，别一进来甩链接。建议路径：

1. 先在通用频道问一个**真的问题**（例如：「你们怎么处理 RWKV 的状态递推在 CPU 上 `exp` 的开销？我用 `exp(-P)` 比 `exp(P)` 慢 2.2×，`reciprocal(exp(P))` 端到端却只在 T=1024 有效」）
2. 有互动之后，再在 `#showcase` / `#research` 类频道发项目，并**明确说**「这是我的项目，想拿 RWKV 当对照基线」

这条「先问问题再展示」的顺序**不是技巧，是基本礼貌**，也直接决定会不会被当广告。

### 2.3 Show HN

标题格式必须是 `Show HN: <作品名> – <一句话说明>`：

```
Show HN: SunnyNet – an O(N) sequence architecture that trains on CPU, no GPU
```

描述第一句就要给**可验证的东西**：

> I built a sequence architecture designed for CPU training (no GPU, no CUDA).
> At T=4096 forward latency is 22.7× a size-matched Transformer, but **it is
> slower than Mamba below T≈384** — the crossover depends only on sequence
> length, not model size. Full measurements, including the losses:
> https://github.com/SunnyClaw-cx/SunnyNet

HN 点：**CPU-only** 是这里的钩子；**主动认输的 crossover** 是可信度来源。

### 2.4 r/MachineLearning

- **不要**发主 feed，先找当周 `[D] Self-Promotion Thread` 置顶帖
- 若要在主 feed 发，用 `[P]` flair，且必须是**开源发布 + 基准数据 + 局限**三件套
- 格式：

```
[P] SunnyNet: an O(N) linear-recurrence architecture for CPU training

Approach: first-order linear recurrence with pairwise intra-group competition
gating (sigmoid(g_i-g_j)·sigmoid(g_i-g_k)) instead of softmax; depthwise causal
conv + 3-tap circulant state filter + residual stream.

Results (size-matched ~365k compute-stack params, isolated subprocess, median):
  step time T=256: 0.0953 ms/tok vs RWKV 0.0982, Mamba 0.1292
  step time T=512: 0.0986 vs RWKV 0.0951
  600-step ppl:    62.37 vs RWKV 63.86, H3 53.93

Limitations: parity holds only at L=4 (L=6 is 30-40% slower); best point is
T=256; quality trades down vs the wider dim=128 config (54.89 -> 62.37);
single implementation, lr not per-arch tuned, single seed for peers.

Code: https://github.com/SunnyClaw-cx/SunnyNet (Apache-2.0)
```

这个格式之所以在 r/MachineLearning 有效，是因为它**把主持人最想砍的点自己先说完了**。

---

## 3. 节奏（不要一天全发）

| 时间 | 动作 |
|---|---|
| D0 | 开**自己仓库**的 Discussions，发首发帖（`docs/DISCUSSIONS_FIRST_POST.md`） |
| D0 | 补 PR：`Awesome-LA-Papers` / `Awesome-state-space-models` / `aisys-building-blocks` 各加一条目（**三份 PR 描述都不许一样**） |
| D+2 | 在 mamba `Show and tell` 发帖（带真实提问） |
| D+4 | RWKV Discord 先提问，隔天再展示 |
| D+7 | Show HN（工作日上午，美东时间） |
| D+10 | r/MachineLearning 每周自荐帖 |
| D+14 | 知乎长文 / V2EX 分享创造 |

**硬性约束**：
- 同一份正文**不得**在两个社区同时出现
- 同一天**不得**在 ≥3 个社区发同类内容（Reddit 明确把同日跨版同文判为 spam）
- 每帖发出后**至少守评论区 2 小时**，早期互动速度直接影响排序

---

## 4. 为什么这套比"灌水"有效

对比一下两条路的**实际期望收益**：

| | 灌水/多账号 | 上面这套 |
|---|---|---|
| 短期曝光 | 可能高（若未被检测） | 中 |
| 被判定 spam 的风险 | **高，且连带主账号** | 低 |
| 能否被同行引用/复用 | 不能 | 能 |
| 有没有真实的协作入口 | 无 | Zoology 合成任务 PR、awesome 列表收录 |
| 数据可信度 | 一旦被识破归零 | 越追问越站得住 |

你的项目**已经有的优势**恰恰是最难伪造的那一类：诚实的不利结果、可复跑的 JSON、
被实测推翻后保留记录的四处设计修正。这些东西放进社区是资产；用机器人发变成负债。

---

## 5. 发帖前自检（每次）

- [ ] 数字与 `experiments/results/*.json` 逐条核对，**不凭记忆**
- [ ] 「输的部分」整段保留
- [ ] 明示「这是我的项目」（disclose）
- [ ] 该社区的开场和结尾已改写（不照搬）
- [ ] 结尾是一个**真问题**，不是求 star
- [ ] 无 marketing 形容词（revolutionary / 颠覆 / 全面领先 / SOTA）
- [ ] 英文稿无机翻腔