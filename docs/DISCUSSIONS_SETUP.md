# 在 SunnyClaw-cx/SunnyNet 开启 Discussions（两种方式）

## 方式 A：我代你创建（需你点一次授权）

1. 打开 https://github.com/login/device
2. 输入授权码：**`4ED0-A890`**
3. 点 "Authorize github" 确认
4. 告诉我"已授权"，我立即：
   - 启用仓库 Discussions
   - 创建分类（如 "Ideas" / "Q&A" / "Show and tell"）
   - 发布 `docs/DISCUSSIONS_FIRST_POST.md` 作为首发帖

> 授权码 15 分钟内有效。若过期，我再生成一个。
> 该授权仅授予 `repo` + `write:discussion` 权限，可随时在
> https://github.com/settings/applications 撤销。

## 方式 B：你自己在网页操作（完全不需要我）

### 1. 启用 Discussions

打开 https://github.com/SunnyClaw-cx/SunnyNet/settings
→ 找到 **Features** 区块 → 勾选 **Discussions** → 保存。

### 2. 新建分类（可选）

https://github.com/SunnyClaw-cx/SunnyNet/discussions/categories
→ 建议保留默认的 `General` / `Ideas` / `Q&A` / `Show and tell`。

### 3. 发首发帖

打开 https://github.com/SunnyClaw-cx/SunnyNet/discussions/new
→ 选 **Show and tell**（或 Announcements）
→ 标题填：

```
SunnyNet 实测对照数据（含不利结果）：20M 质量第一，100M 退回第二
```

→ 正文直接**复制 `docs/DISCUSSIONS_FIRST_POST.md` 的全部内容**粘贴。

### 4. 可选：更新 README 加讨论入口

在 `README.md` 顶部徽章区加一行：

```markdown
[![Discussions](https://img.shields.io/github/discussions/SunnyClaw-cx/SunnyNet)](https://github.com/SunnyClaw-cx/SunnyNet/discussions)
```

## 为什么首发帖里保留了"输的部分"

技术社区（尤其中文技术圈）对"全面领先"式宣传高度警惕。
这份帖子的可信度**恰恰来自它承认**：

- 训练步仍比 Mamba/H3 慢 8–37%
- 100M 规模质量退回第二（落后 rwkv 3.6%）
- 0.5B 只有 SGD 单步能跑，AdamW OOM
- hyena 在本机不稳定，**没有编它的数值**

这些都是可复核的（对应 `experiments/results/*.json`），
比一个漂亮但经不起追问的数字更有说服力。

## 对外发帖（可选的第二步，建议由你本人做）

`docs/COMMUNITY_POST_DRAFT.md` 是给外部社区的中英双语稿。
**请不要**用脚本批量发到 Mamba/RWKV/H3 等他人仓库——那是 spam，
会被举报并可能连累账号。正当做法是：

- 在**自己仓库**的 Discussions 发布（即方式 A/B）
- 若要在他人社区讨论，以**真实技术问题**开场（例如"我的手写 CPU 对照实现
  比官方慢，想确认是不是实现问题"），而不是"来看我的项目"
- 技术自荐的正规渠道：Hacker News (Show HN)、Reddit r/MachineLearning、
  知乎、V2EX