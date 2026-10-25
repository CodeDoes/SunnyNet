# 模型输入 → 输出 示例

本文件收录 SunnyNet 各训练模型**实际生成**的输入 → 输出样本与对应统计。
所有样本都来自真实 checkpoint 的自回归解码，不手写、不润色；能追溯到的原始
日志/命令都在文末列出。生成参数默认 `--temperature 0.7`（采样）或 `--temperature 0`
（贪心 argmax），`--max-new` 见各行。

> **读法**：这些模型处在「流畅 ≠ 连贯」区——词形与节奏像英语/像 Python，
> 但内容不成立。这是 ~1 token/param 欠训练 + 94M 对 35 本书 5.5 过拟合的正常
> 表现，**不是架构缺陷**。§5 用数字证明这一点。

## 1. 本地 94M `p100m_en.pt`（35 本经典, PPL-in-domain 11.9）

checkpoint `runs/p100m_en.pt` + 其训练时 tokenizer `runs/p100m_en_tokenizer.json`，
本机 GPU/CPU 实时生成。

| 输入 prompt | 温度 | 输出（截断） |
|---|---|---|
| `It was the best of times, ` | 0.7 | "had you wreet the hall carriage in the other garden, to the finished the stranching, where he was since the severend girls in the reshring his" |
| `Once upon a time,` | 0.7 | "brought before this corner-clock and a water in my soul back with a body" |
| `def fibonacci(n):` | 0.7 | "and the slaw do not work to be want to be a little ready came to any wh" |
| `The cat sat on the mat` | 0 (贪心) | "ter of the world of the world of the world of the world of the wor" |
| `It is a beautiful day` | 0 (贪心) | "and the world of the world of the world of the world of the world" |

**注意贪心(§1 末两行)**：temp 0 时模型塌缩进「the world of the world of …」
的重复循环——这是 35 本书 5.5 过拟合、把某段高频短语记死后的典型退化，
直接证据见 §5。

## 2. P100 多样化散文 `p100m_en_024000.pt`（197 本书, ~1.5 epoch）

Kaggle P100 实测，存档于 `kaggle_out_p100/prose.final.json`，temp 0.7：

| 输入 prompt | 输出（截断） |
|---|---|
| `It was the best of times, ` | "and I heard at the believe and On a true would have been the laugh of a few follow on it windo" |
| `Once upon a time, ` | "then in the twenty back I know rejected to the world of the Church Parrence in the stroving of the" |
| `The morning sun was ` | "declared to the since more be banked by the corporation and would be be sure presented her care" |

（同一日志里 milestone 000/12000 的样本质量更差、更接近随机；24000 是末段。）

## 3. P100 70/30 散文+代码 `p100m_code_026000.pt`（94M）

Kaggle P100 实测，存档于 `kaggle_out_p100/codemix.final.json`，temp 0.7。
模型学会**代码的外形**（缩进、`for i in range(…)`、`# TODO` 注释形），但产出
碎片而非完整函数：

| 输入 prompt | 输出（截断） |
|---|---|
| `def quicksort(arr):` | '"marks["][80] * [""i, "", "eq")   " / 10_scalar + 1, "0])' |
| `import re` | "for i in range(response):" |
| `# TODO:` | "F \| None, _vike a merely particularly. For the same of Chan gives and has been a gre" |

## 4. 21M `proto20m`（混合域, 仅 1k–10k 步）

`runs/proto20m.pt`（中英+代码混合）只有 ~1k 步、`runs/proto20m_en.pt`（纯英文）
~10k 步，都远未收敛：

| checkpoint | 输入 | 输出 |
|---|---|---|
| proto20m（混合, 1k 步） | `def quicksort(arr):` | "o[ ？ o ] x) … ur he.p()k." |

（混合 1k 步模型连英文都未成句；这是预期的欠训练态，保留仅作对照。）

## 5. 这些输出到底说明什么（数字）

把 §1 的 94M 模型放到**没见过的书**上测（本机实测，tokenizer 用模型自带的），
对照无模型的一元词频下界：

| 语料 | PPL | top-1 下一词准确率 |
|---|---:|---:|
| 域内（它训练过的 35 本书） | **12.0** | 43.1% |
| 域外（2 本从未见过的书） | **442** | 8.1% |
| 一元词频（无模型，94M 分词器） | 79.2 | — |

- 域外 PPL **比「只是背了词频」还差 5.6×**（442 vs 79）。那个漂亮的
  「PPL 11.9」是**域内/自我记忆**的拟合度，不是语言质量。
- 21M `proto20m_en` 域外 PPL ≈ 18,632（top-1 11.9%）——只见过 ~20M 混合 token，
  OOD 基本随机，符合预期。
- 两个 P100 的 val 曲线**单调降到最后一步**（train 一直比 val 低 ~0.1），
  **没有**发散、没有过拟合尖峰——排除「训练/优化器/kernel 出错」的可能。

**结论**：学习没坏；坏在「一直拿域内 PPL 当质量数字报」。真正的判据应是
域外 PPL + top-1，且 94M@35 本书是 5.5 epoch 过拟合。唯一有望跳出该区间的是
`kaggle/big_data/` 内核（21M、~1000 本书、seq 1024、~12 tok/param）。

## 6. 复现

```sh
# §1 本地 94M（需 runs/p100m_en.pt + 其 tokenizer）
python -m sunnynet.cli --ckpt runs/p100m_en.pt \
  --tokenizer runs/p100m_en_tokenizer.json \
  --prompt "It was the best of times, " --max-new 60 --temperature 0.7

# 贪心退化复现（§1 末两行）
python -m sunnynet.cli --ckpt runs/p100m_en.pt \
  --tokenizer runs/p100m_en_tokenizer.json \
  --prompt "The cat sat on the mat" --max-new 40 --temperature 0

# §2/§3 P100 存档样本（无需重训，直接解析日志）
python kaggle_out_p100/parse_logs.py

# §5 域外 PPL / top-1 对照
python kaggle_out_p100/mem_test.py      # 94M 域内 vs 域外
python kaggle_out_p100/ood_compare.py    # 94M vs 21M vs 一元下界
```

### 数据来源

| 内容 | 位置 |
|---|---|
| §1/§4 本地 checkpoint | `runs/*.pt`（gitignore，仅本机） |
| §2 散文 P100 日志 | `kaggle_out_p100/prose.final.json` |
| §3 代码 P100 日志 | `kaggle_out_p100/codemix.final.json` |
| §5 实测脚本 | `kaggle_out_p100/mem_test.py`, `kaggle_out_p100/ood_compare.py` |
