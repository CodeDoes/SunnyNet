"""长距检索在当前(修正后)架构上的复测.

背景: §8 的探针 ckpt (data/ck_probe_sunnynet.pt) 实测 state_dict 只含 blocks.N.delta,
无 theta/delta_max/delta_min ⇒ 出自 §12 修正前的旧架构, 当前架构加载不了.
因此 §8 的数值不代表残差旁路 + CSN 地板 + 可学 theta 之后的检索能力, 需复测.

本节做两件事:
  1. 修正后架构上重训 SunnyNet + Transformer 同协议, 跑 long_range_recall
     (code_python / en_literature / zh_ancient, ngram 1/4/8), 与 §8 同口径对照.
  2. 记忆跨度旋钮: h_max 2k/8k/32k 三档, 看 far 桶命中是否随状态保留跨度改善.
     §9 的判断是 "h_max 只保证不被遗忘, 不保证可被检索" —— 本节直接检验它.

用法:
  python -X utf8 experiments/_longrecall.py train <arch> <steps> [h_max]
  python -X utf8 experiments/_longrecall.py eval <ckpt> [seq_len] [ngram] [n_seqs]
  python -X utf8 experiments/_longrecall.py run [steps]
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
CKPT_DIR = ROOT / "experiments" / "ckpts_longrecall"
DOMAINS = ["zh_ancient", "en_literature", "code_python"]

EVAL = r'''
import json, sys
sys.path.insert(0, r"{root}")
import torch
import torch.nn.functional as F
import numpy as np
from build_corpus import DOMAINS
from train.evaluate import load_model, long_range_recall

model, cfg = load_model(r"{ckpt}")
vocab = cfg.vocab_size
T = {T}
buckets = [1, 4, 8] if {ngram} == 0 else [{ngram}]
res = {{"ckpt": r"{ckpt}", "seq_len": T, "long_range": {{}}}}
for n in buckets:
    tag = f"ngram{{n}}"
    res["long_range"][tag] = {{}}
    for d in DOMAINS:
        r = long_range_recall(model, d, T, {n_seqs}, vocab, ngram=n)
        res["long_range"][tag][d] = r
print("EVAL " + json.dumps(res, ensure_ascii=False))
'''


def run_train(arch: str, steps: int, h_max: float) -> str:
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    tag = arch if h_max <= 0 else f"{arch}_h{int(h_max)}"
    ckpt = CKPT_DIR / f"{tag}.pt"
    out = ROOT / "experiments" / "results" / f"_longrecall_train_{tag}.json"
    cmd = [PY, "-X", "utf8", str(ROOT / "train" / "train.py"),
           "--arch", arch, "--preset", "small", "--steps", str(steps),
           "--batch-size", "4", "--seq-len", "1024", "--lr", "3e-3",
           "--warmup", "60", "--seed", "1234", "--threads", "8",
           "--val-every", str(steps), "--val-batches", "4", "--no-early-stop",
           "--out", str(out), "--save-ckpt", str(ckpt)]
    if h_max > 0:
        cmd += ["--h-max", str(h_max)]
    subprocess.run(cmd, cwd=str(ROOT), check=False)
    return str(ckpt)


def run_eval(ckpt: str, T: int = 1024, ngram: int = 0, n_seqs: int = 6) -> dict:
    code = EVAL.format(root=str(ROOT).replace("\\", "\\\\"), ckpt=ckpt.replace("\\", "\\\\"),
                       T=T, ngram=ngram, n_seqs=n_seqs)
    r = subprocess.run([PY, "-X", "utf8", "-c", code], capture_output=True,
                       text=True, encoding="utf-8")
    line = [ln for ln in r.stdout.splitlines() if ln.startswith("EVAL ")]
    if not line:
        print(f"  EVAL FAILED: {r.stderr[-500:]}")
        return {}
    return json.loads(line[0][5:])


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        print(run_train(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 600,
                        float(sys.argv[4]) if len(sys.argv) > 4 else 0.0))
        return
    if len(sys.argv) > 1 and sys.argv[1] == "eval":
        ck = sys.argv[2]
        T = int(sys.argv[3]) if len(sys.argv) > 3 else 1024
        ng = int(sys.argv[4]) if len(sys.argv) > 4 else 0
        ns = int(sys.argv[5]) if len(sys.argv) > 5 else 6
        print(json.dumps(run_eval(ck, T, ng, ns), ensure_ascii=False))
        return

    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    jobs = [("sunnynet", 0.0), ("transformer", 0.0),
            ("sunnynet", 8192.0), ("sunnynet", 32768.0)]
    rows = []
    for arch, hm in jobs:
        tag = arch if hm <= 0 else f"{arch}_h{int(hm)}"
        print(f"--- train {tag} steps={steps} T=1024 ---")
        ck = run_train(arch, steps, hm)
        if not Path(ck).exists():
            print("    无 ckpt"); continue
        r = run_eval(ck)
        if not r:
            continue
        r["tag"] = tag
        rows.append(r)
        for n in (1, 4, 8):
            t = r["long_range"][f"ngram{n}"]
            line = " ".join(f"{d.split('_')[0]}={t[d].get('far', {}).get('acc')}"
                            for d in DOMAINS)
            print(f"    ngram{n} far: {line}")

    out = ROOT / "experiments" / "results" / "_longrecall.json"
    out.write_text(json.dumps({"steps": steps, "rows": rows},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n写出 {out}")


if __name__ == "__main__":
    main()