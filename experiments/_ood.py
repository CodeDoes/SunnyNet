"""跨分布 (OOD) 泛化实测: 原三域训练 -> 新独立数据集测试.

动机: 质疑"换独立数据集效果差很多". 既有 600 步对照只报**同分布**验证 ppl,
从未做过 "训练分布 vs 未见分布" 的落差测量. 本脚本补这一格.

关键是**测试集必须是训练时从未见过的新数据集** (data/raw_ext/):
  arxiv_en      英文科学摘要 (arxiv API)
  zh_tech_news  简体中文科技新闻 (新华网/光明网/人民网科技)

口径: 同协议训练 (原三域混合 0.4/0.35/0.25, 600 步, batch 8, T 256, lr 3e-3,
seed 1234), 存 ckpt; 再在 "原三域 val + 新域 val" 上算 ppl, 报
ood_gap = 新域 ppl / 原域 ppl 均值.

重要公平性说明: zh_tech_news 的未登录字符率约 23% (BPE 词表来自繁体古籍,
简体新闻大量落表外), arxiv_en 仅 0.05%. 因此中文新域的高 ppl **不全是分布
偏移**, 有分词器不适配的成分 —— 报告须分开陈述, 不能只看一个比值.

用法:
  python -X utf8 experiments/_ood.py train <arch> [steps]
  python -X utf8 experiments/_ood.py eval <ckpt>
  python -X utf8 experiments/_ood.py run [steps] [archs]
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
CKPT_DIR = ROOT / "experiments" / "ckpts_ood"
ORIG = ["zh_ancient", "en_literature", "code_python"]
EXT = ["arxiv_en", "zh_tech_news"]

EVAL = r'''
import json, math, sys
sys.path.insert(0, r"{root}")
import numpy as np
import torch
from pathlib import Path
from train.evaluate import load_model

PROC = Path(r"{root}") / "data" / "proc"
PROC_EXT = Path(r"{root}") / "data" / "proc_ext"
model, cfg = load_model(r"{ckpt}")
vocab = 8192
T = {T}

def ppl_on(domain, is_ext):
    base = PROC_EXT if is_ext else PROC
    a = np.fromfile(base / f"val_{{domain}}.bin", dtype=np.uint16)
    if a.size < T + 2:
        return None
    rng = np.random.default_rng(2024)
    n_win = 24
    starts = rng.integers(0, a.size - T - 1, size=n_win)
    tot, n = 0.0, 0
    model.eval()
    with torch.no_grad():
        for s in starts:
            w = a[s:s + T + 1].astype(np.int64)
            x = torch.from_numpy(w[:-1])[None, :]
            y = torch.from_numpy(w[1:])[None, :]
            lg = model(x)
            tot += float(torch.nn.functional.cross_entropy(
                lg.reshape(-1, vocab), y.reshape(-1), reduction="sum"))
            n += y.numel()
    return round(math.exp(min(tot / max(1, n), 20)), 3)

d = torch.load(r"{ckpt}", map_location="cpu", weights_only=False)
res = {{"arch": d.get("arch"), "ppl": {{}}}}
for dom in {orig}:
    res["ppl"][dom] = ppl_on(dom, False)
for dom in {ext}:
    res["ppl"][dom] = ppl_on(dom, True)
o = [v for k, v in res["ppl"].items() if k in {orig} and v]
e = [v for k, v in res["ppl"].items() if k in {ext} and v]
res["orig_mean"] = round(sum(o) / len(o), 3) if o else None
res["ext_mean"] = round(sum(e) / len(e), 3) if e else None
if res["orig_mean"] and res["ext_mean"]:
    res["ood_gap"] = round(res["ext_mean"] / res["orig_mean"], 3)
print("EVAL " + json.dumps(res, ensure_ascii=False))
'''


def run_train(arch: str, steps: int) -> str:
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    out = ROOT / "experiments" / "results" / f"_ood_train_{arch}.json"
    ckpt = CKPT_DIR / f"{arch}.pt"
    cmd = [PY, "-X", "utf8", str(ROOT / "train" / "train.py"),
           "--arch", arch, "--preset", "small", "--steps", str(steps),
           "--batch-size", "8", "--seq-len", "256", "--lr", "3e-3",
           "--warmup", "60", "--seed", "1234", "--threads", "8",
           "--val-every", str(steps), "--val-batches", "8", "--no-early-stop",
           "--out", str(out), "--save-ckpt", str(ckpt)]
    subprocess.run(cmd, cwd=str(ROOT), check=False)
    return str(ckpt)


def run_eval(ckpt: str, T: int = 1024) -> dict:
    code = EVAL.format(root=str(ROOT).replace("\\", "\\\\"), ckpt=ckpt, T=T,
                       orig=repr(ORIG), ext=repr(EXT))
    r = subprocess.run([PY, "-X", "utf8", "-c", code], capture_output=True,
                       text=True, encoding="utf-8")
    line = [ln for ln in r.stdout.splitlines() if ln.startswith("EVAL ")]
    if not line:
        print(f"  EVAL FAILED: {r.stderr[-400:]}")
        return {}
    return json.loads(line[0][5:])


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        print(run_train(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 600))
        return
    if len(sys.argv) > 1 and sys.argv[1] == "eval":
        print(json.dumps(run_eval(sys.argv[2],
                                  int(sys.argv[3]) if len(sys.argv) > 3 else 1024),
                         ensure_ascii=False))
        return

    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    archs = sys.argv[2].split(",") if len(sys.argv) > 2 else ["sunnynet", "transformer"]
    rows = []
    for arch in archs:
        print(f"--- train {arch} 原三域 {steps} 步 ---")
        ck = run_train(arch, steps)
        if not Path(ck).exists():
            print("    无 ckpt"); continue
        r = run_eval(ck)
        if not r:
            continue
        rows.append(r)
        print(f"    ppl={r['ppl']}  orig={r['orig_mean']} ext={r['ext_mean']} "
              f"ood_gap={r.get('ood_gap')}")

    out = ROOT / "experiments" / "results" / "_ood.json"
    out.write_text(json.dumps({"steps": steps, "orig": ORIG, "ext": EXT,
                               "rows": rows}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\n=== OOD 泛化 (原三域训练 {steps} 步, 测试 T=1024) ===")
    print("arch".ljust(13) + "".join(d.ljust(14) for d in ORIG + EXT) +
          "orig_mean".ljust(12) + "ext_mean".ljust(12) + "ood_gap")
    for r in rows:
        print(r["arch"].ljust(13) +
              "".join(str(r["ppl"].get(d)).ljust(14) for d in ORIG + EXT) +
              f"{r['orig_mean']:<12}" + f"{r['ext_mean']:<12}" +
              f"{r.get('ood_gap')}")
    print(f"\n写出 {out}")


if __name__ == "__main__":
    main()