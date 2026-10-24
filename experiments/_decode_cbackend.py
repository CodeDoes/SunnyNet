"""解码态 C 整模型后端 vs torch eager 的同规格对照.

§23.3 的解码记分卡走 torch forward_step; 本脚本在同规格 (d80/i160/r244/L4,
与 §23.3 的 sunny 格完全一致) 下补 C 后端一格, 判定名次是否改变.

口径与 §23.3 对齐: 随机初始化, 每格独立子进程, 单线程, 先走满 T 步建立状态再测单步。
"""
from __future__ import annotations

import json
import os
import statistics as st
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
SPEC = (80, 160, 244, 4)
V = 8192
CKPT = ROOT / "experiments/ckpts_decode/d80.pt"
BLOB = ROOT / "experiments/ckpts_decode/d80.runtime.bin"

CELL = r'''
import json, sys, time
sys.path.insert(0, r"{root}")
import torch, statistics as st
T, kind, reps = {T}, "{kind}", {reps}
torch.set_num_threads(1); torch.manual_seed(1234)
from sunnynet.cli import load_model
m, cfg, _ = load_model(r"{ckpt}")
m.eval()
if kind == "c":
    from sunnynet.runtime.model import SnModel
    rt = SnModel(r"{blob}", cfg.vocab_size, cfg.dim, cfg.inner, cfg.rank,
                 cfg.taps, cfg.layers, 8, int(cfg.tie_embeddings))
idx = torch.randint(0, cfg.vocab_size, (1, T))
if kind == "c":
    rt.prefill(idx[0].tolist())
    one = int(torch.randint(0, cfg.vocab_size, (1,)).item())
    rt.step(one)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); rt.step(one); ts.append(time.perf_counter()-t0)
else:
    s = m.init_state(1)
    with torch.no_grad():
        for t in range(T):
            _, s = m.forward_step(idx[:, t], s)
        one = torch.randint(0, cfg.vocab_size, (1,))
        m.forward_step(one, s)
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter(); m.forward_step(one, s); ts.append(time.perf_counter()-t0)
print("CELL " + json.dumps({{"ms": st.median(ts) * 1000}}))
'''

EQUIV = r'''
import json, sys
sys.path.insert(0, r"{root}")
import torch
torch.set_num_threads(1)
from sunnynet.cli import load_model
from sunnynet.runtime.model import SnModel
m, cfg, _ = load_model(r"{ckpt}")
m.eval()
rt = SnModel(r"{blob}", cfg.vocab_size, cfg.dim, cfg.inner, cfg.rank,
             cfg.taps, cfg.layers, 8, int(cfg.tie_embeddings))
torch.manual_seed(7)
toks = torch.randint(0, cfg.vocab_size, (48,)).tolist()
s = m.init_state(1); mx = 0.0; hit = 0
with torch.no_grad():
    for t in toks:
        lg_t, s = m.forward_step(torch.tensor([t]), s)
        lg_c = rt.step(t)
        a = lg_t[0].numpy(); b = torch.tensor(lg_c).numpy()
        mx = max(mx, float(abs(a - b).max()))
        hit += int(a.argmax()) == int(b.argmax())
print("EQUIV " + json.dumps({{"maxdiff": mx, "argmax": f"{{hit}}/{{len(toks)}}"}}))
'''


def reps_for(T):
    return 64 if T <= 1024 else (32 if T <= 2048 else 16)


def make_ckpt():
    import torch
    from sunnynet.model import SunnyConfig, SunnyNet
    torch.manual_seed(1234)
    c = SunnyConfig(vocab_size=V, dim=SPEC[0], inner=SPEC[1], rank=SPEC[2],
                    layers=SPEC[3], h_max=2048.0)
    c.scan_chunk = 64
    m = SunnyNet(c)
    CKPT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"arch": "sunnynet", "model": m.state_dict(),
                "config": c.__dict__ if hasattr(c, "__dict__") else c},
               CKPT)
    need = any(b._need_clamp for b in m.blocks)
    if need:
        raise SystemExit("该规格需要 clamp, C 后端不支持")
    print(f"[ckpt] {CKPT}  params={m.num_params(non_embedding=True):,}")


def export_blob():
    from experiments._export_runtime import export
    info = export(str(CKPT), str(BLOB))
    print(f"[blob] {BLOB}  {info['bytes']/1e6:.2f} MB")


def run_child(code):
    r = subprocess.run([PY, "-X", "utf8", "-c", code], capture_output=True,
                       text=True, encoding="utf-8", cwd=str(ROOT),
                       env=dict(os.environ))
    for ln in r.stdout.splitlines():
        for tag in ("CELL ", "EQUIV "):
            if ln.startswith(tag):
                return json.loads(ln[len(tag):])
    raise RuntimeError((r.stderr or r.stdout or "")[-1200:])


def main():
    sys.path.insert(0, str(ROOT))
    if not CKPT.exists():
        make_ckpt()
    if not BLOB.exists():
        export_blob()

    eq = run_child(EQUIV.format(root=str(ROOT), ckpt=CKPT, blob=BLOB))
    print(f"[equiv] 单步 logits max|diff| {eq['maxdiff']:.2e}  argmax {eq['argmax']}")

    Ts = [int(x) for x in (sys.argv[1].split(",") if len(sys.argv) > 1
                           else ["256", "512", "1024", "2048", "4096"])]
    rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    acc = {(k, T): [] for k in ("torch", "c") for T in Ts}
    for rd in range(rounds):
        for T in Ts:
            for k in ("torch", "c"):
                v = run_child(CELL.format(root=str(ROOT), T=T, kind=k,
                                          reps=reps_for(T), ckpt=CKPT, blob=BLOB))["ms"]
                acc[(k, T)].append(v)
                print(f"  r{rd+1} {k:5s} T={T:<6d} {v:.4f} ms")
    print("\n=== 单步解码 ms 中位 (C 整模型后端 vs torch eager) ===")
    tbl = {"torch": [], "c": []}
    for T in Ts:
        for k in ("torch", "c"):
            tbl[k].append(st.median(acc[(k, T)]))
    print("T".ljust(8) + "".join(f"T={T}".ljust(12) for T in Ts))
    for k in ("torch", "c"):
        print(f"  {k:6s}" + "".join(f"{v:.4f}".ljust(12) for v in tbl[k]))
    print("  比值  " + "".join(f"{t/c:.3f}".ljust(12) for t, c in zip(tbl["torch"], tbl["c"])))
    out = ROOT / "experiments/results/_decode_cbackend.json"
    out.write_text(json.dumps({"Ts": Ts, "table": tbl, "equiv": eq, "rounds": rounds},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n写出 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())