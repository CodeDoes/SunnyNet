"""递推状态的动态范围与量化误差实测.

论点拆成两条, 分开验证:
  A. 动态范围: 凸组合状态 |s_t| <= max(|s_0|, max|u|) 由构造锁定, 不随 T 漂移;
     无界累加 (a=1) 的幅度随 T 线性增长 —— 这决定量化标度是否稳定。
  B. 量化误差: 状态量化的真实用法是跨段/跨块传递时压缩 (一次),
     而非每步来回量化。两种用法分开测。

口径: 真实 ckpt 第 0 层前向的 p/u 序列 (B=1); fp32 参照; 对称 INT8;
      每格独立子进程。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "experiments" / "results" / "_state_quant.json"
PY = sys.executable
CKPT = ROOT / "experiments" / "ckpts_ood" / "sunnynet.pt"


def collect_pu(T: int):
    from sunnynet.model import SunnyConfig, SunnyNet
    d = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg = SunnyConfig(**d["config"])
    m = SunnyNet(cfg)
    m.load_state_dict(d["model"])
    m.eval()
    blk = m.blocks[0]
    torch.manual_seed(0)
    idx = torch.randint(0, cfg.vocab_size, (1, T))
    with torch.no_grad():
        x = m.embed(idx)
        raw = blk.norm_in(x)
        a = blk._apply_local(raw, torch.zeros(1, blk.taps - 1, cfg.dim))
        u, gate, p = blk._core(a)
    return p[0].contiguous(), u[0].contiguous()


def qrt(x, block):
    from sunnynet.runtime import quant_i8, dequant_i8
    q, sc = quant_i8(x, block)
    return dequant_i8(q, sc, block)


def run_case(T, block, mode, quant_mode):
    """quant_mode: once (跨段传递, 只量化一次) / every (每步)。"""
    p, u = collect_pu(T)
    D = p.shape[-1]
    a_seq = p if mode == "bounded" else torch.ones_like(p)

    # fp32 参照
    s_ref = torch.zeros(D)
    ref_absmax = []
    for t in range(T):
        if mode == "bounded":
            s_ref = a_seq[t] * s_ref + (1.0 - a_seq[t]) * u[t]
        else:
            s_ref = s_ref + u[t]          # 无界累加 (未归一 KV)
        ref_absmax.append(s_ref.abs().max().item())

    # 量化递推
    s = torch.zeros(D)
    errs = []
    for t in range(T):
        if mode == "bounded":
            s = a_seq[t] * s + (1.0 - a_seq[t]) * u[t]
        else:
            s = s + u[t]
        if quant_mode == "every":
            s = qrt(s.view(1, D), block).view(D)
        errs.append((s - s_ref).abs().max().item())
    if quant_mode == "once":
        s = qrt(s.view(1, D), block).view(D)
        errs = [(s - s_ref).abs().max().item()]

    drift = ref_absmax[-1] / max(ref_absmax[T // 8], 1e-12)
    return {
        "T": T, "block": block, "mode": mode, "quant_mode": quant_mode, "D": D,
        "absmax_first": ref_absmax[T // 8], "absmax_last": ref_absmax[-1],
        "absmax_drift": drift,
        "err_final": errs[-1],
        "err_ratio": errs[-1] / max(ref_absmax[-1], 1e-12),
        "used_bits": 8,
    }


def run_fixed_scale(T, mode):
    """固定标度: 用序列 1/2 处的 absmax 定 8bit 标度, 量化末态。

    这是"标度不重算"的部署用法。状态幅度漂移超出标度即溢出截断。
    """
    p, u = collect_pu(T)
    D = p.shape[-1]
    s = torch.zeros(D)
    half = T // 2
    scale = None
    for t in range(T):
        if mode == "bounded":
            s = p[t] * s + (1.0 - p[t]) * u[t]
        else:
            s = s + u[t]
        if t == half:
            scale = s.abs().max().item() / 127.0
    # 量化末态
    q = torch.clamp(torch.round(s / scale), -127, 127)
    s_q = q * scale
    overflow = int((torch.round(s / scale).abs() > 127).sum().item())
    return {
        "T": T, "mode": mode, "D": D,
        "scale_from_half": scale,
        "absmax_end": s.abs().max().item(),
        "overflow_ch": overflow,
        "overflow_ratio": overflow / D,
        "rel_err": (s_q - s).abs().max().item() / max(s.abs().max().item(), 1e-12),
    }


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        T, block, mode, qm = int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5]
        print(json.dumps(run_case(T, block, mode, qm)))
        return 0
    if len(sys.argv) > 1 and sys.argv[1] == "--fixed":
        T, mode = int(sys.argv[2]), sys.argv[3]
        print(json.dumps(run_fixed_scale(T, mode)))
        return 0
    cfgs = [
        (2048, 0, "bounded", "once"),
        (2048, 0, "unbounded", "once"),
        (2048, 32, "bounded", "once"),
        (2048, 0, "bounded", "every"),
        (2048, 0, "unbounded", "every"),
    ]
    rows = []
    fixed = []
    env = {**os.environ, "SUNNY_W64": "F:/devtools/w64devkit"}
    for T, block, mode, qm in cfgs:
        r = subprocess.run([PY, "-X", "utf8", str(Path(__file__).resolve()),
                            "--child", str(T), str(block), mode, qm],
                           capture_output=True, text=True, cwd=str(ROOT), env=env)
        try:
            rows.append(json.loads(r.stdout.strip().splitlines()[-1]))
        except Exception:
            print(f"  失败 {mode} {qm}: {(r.stderr or r.stdout or '')[-200:]}")
    for mode in ("bounded", "unbounded"):
        r = subprocess.run([PY, "-X", "utf8", str(Path(__file__).resolve()),
                            "--fixed", "2048", mode],
                           capture_output=True, text=True, cwd=str(ROOT), env=env)
        try:
            fixed.append(json.loads(r.stdout.strip().splitlines()[-1]))
        except Exception:
            print(f"  失败 fixed {mode}: {(r.stderr or r.stdout or '')[-200:]}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"recursive": rows, "fixed_scale": fixed},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{'模式':>10} {'量化':>6} {'块':>4} {'|s|首':>9} {'|s|末':>10} "
          f"{'漂移':>7} {'末误差':>10} {'相对':>9}")
    print("-" * 70)
    for d in rows:
        print(f"{d['mode']:>10} {d['quant_mode']:>6} {d['block']:>4} "
              f"{d['absmax_first']:>9.3f} {d['absmax_last']:>10.3f} "
              f"{d['absmax_drift']:>6.2f}x {d['err_final']:>10.2e} {d['err_ratio']:>9.2e}")
    print("-" * 70)
    print(f"{'固定标度':>10} {'|s|末':>10} {'溢出通道':>10} {'溢出比':>9} {'相对误差':>10}")
    print("-" * 52)
    for d in fixed:
        print(f"{d['mode']:>10} {d['absmax_end']:>10.2f} {d['overflow_ch']:>10d} "
              f"{d['overflow_ratio']:>8.2%} {d['rel_err']:>10.2e}")
    print("-" * 52)
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())