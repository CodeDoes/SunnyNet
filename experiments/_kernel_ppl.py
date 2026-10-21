"""内核路 vs torch 路的 ppl 与训练曲线对照.

内核与 torch 参考数值逐位等价 (loss 已实测 0.000e+00), 故 ppl 应当一致;
本脚本实测确认, 并同时给出训练步耗时与 token/s, 用于"训练速度与质量"的联合陈述.

两路各自独立子进程, 同协议 (seed/步数/lr/数据混比全一致)。

用法: python -X utf8 experiments/_kernel_ppl.py [steps] [preset] [T]
"""
from __future__ import annotations
import json, os, statistics as st, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

CELL = r'''
import json, os, sys, time
sys.path.insert(0, r"{root}")
os.environ["SUNNY_KERNEL"] = "{flag}"
import torch
from train.train import main as _unused  # noqa
'''

RUNNER = r'''
import json, os, sys, time
sys.path.insert(0, r"{root}")
import argparse, torch
from train.train import build_models, DomainMixer, DEFAULT_MIX
from build_corpus import load_tokenizer
from sunnynet.kernels.core_auto import set_kernel_active, kernel_active

steps, preset, T, V_bs = {steps}, "{preset}", {T}, {bs}
torch.manual_seed(1234)
torch.set_num_threads({threads})
set_kernel_active({pyflag}, threads={threads})
tk = load_tokenizer(); vocab = tk.vocab_size
sunny, base, cfg = build_models("sunnynet", vocab, preset, None, T, 1234, 0.0)
m = sunny
mx = DomainMixer(dict(DEFAULT_MIX), T, V_bs, "train", seed=1234)
opt = torch.optim.AdamW(m.parameters(), lr=3e-3, betas=(0.9, 0.95), weight_decay=0.01)
import torch.nn.functional as F
hist = []
t0 = time.perf_counter()
for s in range(1, steps + 1):
    x, y = mx.batch()
    logits = m(x)
    loss = F.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1))
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    opt.step()
    hist.append(float(loss.detach()))
wall = time.perf_counter() - t0
# 末段平均 loss -> ppl
tail = hist[-max(1, steps // 10):]
mean_loss = sum(tail) / len(tail)
ppl = float(torch.exp(torch.tensor(mean_loss)))
print("CELL " + json.dumps({{"kernel": kernel_active(), "steps": steps,
      "wall_s": round(wall, 2), "ms_per_step": round(wall / steps * 1000, 3),
      "tokens_per_s": round(steps * V_bs * T / wall, 1),
      "loss_first": round(hist[0], 5), "loss_last": round(hist[-1], 5),
      "tail_mean_loss": round(mean_loss, 5), "ppl": round(ppl, 4)}}))
'''


def run(pyflag, steps, preset, T, bs, threads):
    code = RUNNER.format(root=str(ROOT), pyflag=str(pyflag), steps=steps,
                         preset=preset, T=T, bs=bs, threads=threads)
    env = dict(os.environ); env["SUNNY_W64"] = env.get("SUNNY_W64", r"F:/devtools/w64devkit")
    r = subprocess.run([PY, "-X", "utf8", "-c", code], capture_output=True, text=True,
                       encoding="utf-8", cwd=str(ROOT), env=env)
    for ln in r.stdout.splitlines():
        if ln.startswith("CELL "):
            return json.loads(ln[5:])
    raise RuntimeError(f"runner 失败 (kernel={pyflag}):\n{r.stdout[-1000:]}\n{r.stderr[-1000:]}")


steps = int(sys.argv[1]) if len(sys.argv) > 1 else 200
preset = sys.argv[2] if len(sys.argv) > 2 else "smalln80"
T = int(sys.argv[3]) if len(sys.argv) > 3 else 256
bs, threads = 8, 8
print(f"内核 vs torch: preset={preset} steps={steps} T={T} bs={bs} threads={threads}")
res = {}
for flag in (False, True):
    r = run(flag, steps, preset, T, bs, threads)
    res["kernel" if flag else "torch"] = r
    print(f"  {'kernel' if flag else 'torch ':6s} ppl={r['ppl']:.4f} "
          f"ms/step={r['ms_per_step']:.2f} tok/s={r['tokens_per_s']:.0f} "
          f"loss {r['loss_first']:.4f}->{r['loss_last']:.4f}")
d_ppl = abs(res["torch"]["ppl"] - res["kernel"]["ppl"])
d_l = abs(res["torch"]["tail_mean_loss"] - res["kernel"]["tail_mean_loss"])
print(f"\nppl 差 {d_ppl:.4f} | 末段 loss 差 {d_l:.6f}")
print(f"训练步加速 {res['torch']['ms_per_step']/res['kernel']['ms_per_step']:.3f}x")
out = ROOT / "experiments" / "results" / "_kernel_ppl.json"
out.write_text(json.dumps({"preset": preset, "steps": steps, "T": T, "bs": bs,
                           "threads": threads, "res": res,
                           "ppl_absdiff": d_ppl, "tail_loss_absdiff": d_l},
                          ensure_ascii=False, indent=2), encoding="utf-8")
print(f"写出 {out}")
