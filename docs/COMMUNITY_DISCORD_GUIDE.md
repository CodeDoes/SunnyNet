# RWKV Discord 发帖指南（SunnyNet）

> 本文只写**已核实**的事实。凡是我无法登录后确认的东西（例如具体频道名），
> 一律标注"需你自己进去确认"，不编造。

---

## 0. Discord 是什么网站

Discord 是国外的**即时通讯 + 社区平台**（可以理解成 Slack + 语音房 + 轻论坛的混合体）。
它不是博客、不是论坛帖，也不是问答站。它的核心单位是**服务器（Server）**，
服务器内部再细分**频道（Channel）**：文字频道、语音频道、以及"论坛频道（Forum）"。

对开源项目来说，Discord 通常就是**作者本人和核心开发者天天在线的地方**——
比 GitHub Issues 更随意，比邮件列表更实时。RWKV 官方就把 Discord 当作主要社区入口：
`wiki.rwkv.com` 首页社区链接那一栏，第一条就是 **Discord Forum**。

---

## 1. RWKV 服务器的已核实事实

| 项目 | 值 | 来源 |
|---|---|---|
| 邀请链接 | `https://discord.gg/bDSBUMeFpc` | RWKV 官方 wiki + 各仓库 README，多处一致 |
| 服务器名 | RWKV Language Model（& ChatRWKV） | GroupFinds 目录页 2026-10-05 快照 |
| 成员数 | **9,532**（961 在线，约 10%） | 同上，2026-10-05 实测 |
| 成立时间 | 2022-07-01 | 同上 |
| 类型 | 已开启 Discord Community（有公告频道、被 Server Discovery 收录） | 同上 |

**这个规模意味着什么**：9.5k 人的服务器，日常活跃在 10% 左右。也就是说你的帖子
大概会落在**几百人眼前**——比 Reddit 主版少很多，但**里面全是做线性/RNN 架构的人**，
信噪比比任何一个大版都高。不要用"发出去能不能爆"衡量，要用"能不能触达正确的人"衡量。

---

## 2. 关键区别：Discord 不是论坛，不能贴长文

这是最容易翻车的地方。我在 `COMMUNITY_POSTS_READY.md` 里给的那篇 RWKV 稿是
**论坛口径**（GitHub Discussions / Reddit 那种带标题、带表格、能展开的长文）。
**直接粘进 Discord 会被无视**，原因：

1. **Discord 不渲染 Markdown 表格**——表格会变成一堆竖线和减号的乱码。
2. **消息长度有限**（单条 2000 字符），长的会被截成多条刷屏，观感差。
3. **聊天流是滚动式的**，长消息没人往上翻；帖子要在**论坛频道**里发，不是普通文字频道。

### 正确的 Discord 动作是这四种

| 动作 | 用在哪 | 形态 |
|---|---|---|
| **① 自我介绍** | `#introductions` 一类频道 | 2–3 句，说你是谁、在做什么 |
| **② 论坛频道发帖** | 论坛类频道（Forum） | 标题 + 短正文 + 链接，**这就是"发帖"** |
| **③ 群聊里贴一句** | `#general` / `#off-topic` 一类 | 一句话 + 链接，**只在有人聊到相关话题时** |
| **④ 回帖接话** | 别人帖子下 | 回答问题，**不带自己链接**（除非被问） |

> ⚠️ **频道名我无法替你确认**——Discord 的频道列表必须登录后才能渲染，
> 从公网抓不到。你注册了，所以你自己能看见。进去后按下面的特征找：
> - 找 `welcome` / `rules` / `start-here` → 先读规则
> - 找 `introductions` / `intro` → 自我介绍
> - 找名字里带 **showcase / projects / share / forum / research** 的 → **这才是发项目帖的地方**
> - 找 `general` / `chat` / `lounge` → 闲聊，别一上来就发链接
> - 找 `help` / `support` / `questions` → 提问（提问也是推广，见 §4）

---

## 3. 你进服务器后的正确顺序（照着做）

**第 1 步：读规则频道。** 每个 Discord 服务器第一个必读的就是 `#rules`。
注意找两件事：① 有没有 self-promotion / showcase 的专门规定；② 有没有语言分区
（RWKV 社区国际化，通常有英文为主的 `#general` 和别的语种频道）。

**第 2 步：完成验证/选角色。** 很多服务器有验证门（点个 emoji、选角色）。
不做的话你可能**看不到**大部分频道——如果你觉得"频道好少"，八成是没选角色。

**第 3 步：自我介绍。** 一句话就够，别上来就发链接：
```
Hi all — CX here. I work mostly on CPU-first sequence architectures,
currently building an O(N) recurrence and using RWKV as one of my step-time
baselines. Happy to be here.
```

**第 4 步（关键）：先在别人帖子里回一两次话。** 找个你真懂的问题答一句。
这一步不是客套——**Discord 里"新号第一条消息就是自己的项目链接"和广告没有区别**，
而你先答过两句话，再发项目，观感完全不同。

**第 5 步：发项目帖。** 去那个 showcase / forum 类频道，用 §4 的成品。

**第 6 步：守着回话。** Discord 的价值在于**当场对话**。发完就走等于白发；
留在频道里，有人问就答，这才是"热度"真正的来源。

---

## 4. 现成文案（Discord 口径，可直接复制）

### 4.1 项目帖（论坛频道用）

**标题**
```
An O(N) CPU-first recurrence — step time ties RWKV at T=256; where it doesn't
```

**正文**
```
Hi — I've been building an O(N) sequence architecture aimed at CPU-first training,
and RWKV has been one of my main step-time baselines. Sharing the head-to-head
plus one open question about the CPU side of the recurrence.

Repo (Apache-2.0): https://github.com/SunnyClaw-cx/SunnyNet

Step time (fwd+bwd+update), ~365k compute-stack params, isolated subprocess,
5-round median, ms/token:

  T=256   mine 0.0953   RWKV 0.0982
  T=512   mine 0.0986   RWKV 0.0951

Over 9 paired rounds the sign test went 3/9 (T=256) and 4/9 (T=512), so I report
this as a tie, not a win. At T=512 RWKV is ahead.

Two things I'd genuinely like input on:

1. In my scan exp(-P) is 2.2x slower than exp(P), and reciprocal(exp(P)) is
   numerically fine (fp64 rel. err 7.6e-17, fp32 6.6e-08) — but end-to-end it
   only helps at T=1024, not at T=256/512. Do you see the same shape on the
   RWKV side? I suspect small-tensor scheduling dominates below some T, but I
   don't yet have a measurement that separates the two.

2. When I size-matched by compute-stack params, my low-rank design needed a lot
   more depth (10 layers vs 5-7 at 20M). Re-balancing to dim 512->640 / L 10->6
   at equal params gave 12% faster steps and 10.9% better ppl. Was RWKV's depth
   ever tuned this way, or is it fixed by other constraints?

Caveats I'll state up front: parity only holds at L=4 — at L=6 (matching RWKV's
depth) my equal-budget config is 30-40% slower. And the comparison is not
converged (600 steps, one learning rate, single implementation). Reproduce
script and raw JSON are in the repo.
```

> ✅ **为什么这样写**：第一人称、数字全带条件（batch/T/线程/轮数）、
> 主动写"哪里输了"、结尾提**两个真问题**而不是"求 star"。
> Discord 的人最认这种——**你把他们当同行，他们才会把你当同行**。

### 4.2 群聊里的一句话（只在话题相关时用）

有人聊到线性架构速度 / CPU 推理 / 低秩设计时：
```
Related: I measured this on a CPU-first O(N) recurrence — step time ties RWKV at
T=256 and loses 3.7% at T=512 over 9 paired rounds. Numbers + repro script:
https://github.com/SunnyClaw-cx/SunnyNet (mine, Apache-2.0)
```
**要点**：一句话、带链接、**明说"mine"**。

### 4.3 被追问时的回帖模板

```
Yeah, good catch — at L=6 it's 30-40% slower, I'm not claiming otherwise.
The parity only holds at L=4 at equal stack params. That asymmetry is actually
the most interesting part for me: it suggests my bottleneck is op count, not FLOPs
(MAC/token is ~8064 vs Mamba's ~7029). If you have thoughts on reducing op count
per token without losing the state update, I'd love to hear them.
```

---

## 5. 会害到你的行为（一次都别做）

1. **私信发链接**（DM spam）——Discord 里这是最会被举报的行为，可能直接封号。
2. **@全体 / 批量 @人**——除非你是管理员，否则必被踢。
3. **同一个链接在多个频道重复贴**——判为 spam。
4. **假装不是你自己的项目**——一旦被发现，技术可信度归零，且违反各社区规则。
5. **一进来就发链接、发完就走**——见 §3 第 4 步。
6. **用多个账号顶帖**——Discord 的账号关联检测比 GitHub 更严，得不偿失。

---

## 6. 中文渠道（可作为补充，不要当成"多刷几遍"）

RWKV 中国社区（`rwkv.cn`）有独立的 QQ 群，公开可查的有：

| 群 | 号 |
|---|---|
| ChatRWKV 技术研发群 | 325154699 |
| RWKV 技术讨论群 | 224287095 |
| ChatRWKV 水和机器人群 | 597996773 |
| RWKV 第三方微调/训练/lora 综合群 | 439087067 |

来源：RWKV Wiki 中文版（`RWKV-Wiki-CN`）社区交流章节。

> **纪律**：**同一个内容不要在中英文两个渠道重复刷一遍**。要么中文群问中文问题、
> Discord 发英文项目帖（内容不同），要么只发一处。多贴 = spam。

**另**：`rwkv.cn` 有行为准则页，核心要求是"尊重不同主张、接受建设性意见、
注重社区共同诉求而非个人得失"。这条准则本身也说明了**什么人会被这个社区接受**。

---

## 7. 一句话总结

**Discord 不是发广告的地方，是"作者本人天天在线"的地方。**
你能给这里的最大价值不是"我有个项目"，而是**"我在 CPU 侧测到了这些数、
我有个还没想明白的问题"**——后者才是会引来回复的东西。
先把 §3 的六步走完，再用 §4 的稿子发。