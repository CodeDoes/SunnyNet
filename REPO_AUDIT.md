# 仓库缺口审计

结论来自 API / 文件系统 / 干净副本实测，非推断。
干净副本 = `git archive HEAD | tar -x`（等价 CI 检出，无 `data/` 目录）。

## 一、逐条判定

| # | 审计表主张 | 核实结果（证据） | 判定 |
|---|---|---|---|
| 1 | 0 Release / 0 Tag | `api.github.com/.../tags` → `[]`；`/releases` → `[]` | 属实 |
| 2 | 无 CONTRIBUTING | `.github/` 仅 `FUNDING.yml` + `workflows/tests.yml`；CONTRIBUTING / CODE_OF_CONDUCT / SECURITY / Issue / PR 模板全无 | 属实 |
| 3 | 无样例数据 | `git ls-tree -r HEAD` 219 文件，零 `.bin`/`.csv`/`.pt`；`data/` 在 `.gitignore` | 属实 |
| 4 | 无预训练权重 | 同上，无任何权重文件 | 属实 |
| 5 | 零外部验证 | `contributors` = `sunny-cx`(20 commits) + `SunnyClaw-cx`(3 commits)；两账号 `name`/`bio` 均指向同一人 | 属实 |
| 6 | 无 GPU / 无公开基准 | 报告 §15 整节为 GPU 实测：GTX 1060 前向 8.0–15.2×、训练步 9.9–12.5×，并写明无 GPU 侧同行对照 | 失实 |
| 7 | 对照基线自写 | `sunnynet/linear_baselines.py` 确为手写实现 | 属实，但非缺口 |
| 8 | README 无 CI 徽章 | README 仅 1 个 `shields.io` 徽章（爱发电） | 属实 |
| 9 | CI 绿了 | CI 17 次运行全部 FAILURE，历史上从未成功过；badge 曾实测为 `failing` | 完全相反 → 已修复 |

## 二、表中最关键的两点

- 方向写反的那条才是真缺陷：表称"CI 绿了"，实测是 17/17 全红。
  本地测试全绿会掩盖它——本地存在 `data/proc/tokenizer.json`，CI 检出后没有。
- 看起来最唬人的两条反而不成立：#6 有 GPU 验证（只是无同行对照）、
  #7 手写基线是架构研究的标准做法（同行有论文才可复用其官方实现，且"禁止实现退化"要求自己写）。

## 三、表中漏项（推广前更该补）

| 项 | 证据 | 影响 |
|---|---|---|
| 零 GitHub topics | `api.github.com/.../topics` → `names: []` | 直接影响可发现性，推广前第一优先 |
| 无依赖声明 | `pyproject.toml`/`setup.py`/`requirements.txt` 全无 | 安装体验；版本号 `0.1.0` 只在 `sunnynet/__init__.py` |
| Discussions 未开 | `has_discussions: false` | 社区承载入口缺位 |

## 四、已修复并验证

根因：`train/train.py`、`train/evaluate.py` 硬编码 `ROOT/"data"/"proc"/"tokenizer.json"`，
绕开 `build_corpus.PROC`，令测试的 `monkeypatch.setattr(build_corpus,"PROC",…)` 结构性失效。

- 干净副本复现：`2 failed, 52 passed`（`FileNotFoundError` @ `test_smoke_pipeline.py:102`）
- 修复：改用既有的 `build_corpus.load_tokenizer()`；`smoke-train` job 补装 `pytest`
- 验证：干净副本 54 passed · `bare pytest tests/` 54 passed · smoke 2 passed · 3.12 `compileall` 无错
- 结果：推送后官方 badge `failing → passing`（首次全绿）
- 当前测试数已增至架构 59 项 + 冒烟 2 项，仍全绿。