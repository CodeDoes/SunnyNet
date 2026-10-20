"""SunnyNet 速度优化候选筛选: 单块前向+反向耗时对照.

每个候选通过 monkeypatch 改写实现, 在同一进程内 A/B. 用于快速淘汰,
有效候选再进独立子进程端到端验证.

运行: python -X utf8 experiments/_opt_screen.py [--seq-len 1024] [--batch 8] [--threads 8]
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from sunnynet.model import SunnyConfig, SunnyNet  # noqa: E402
from sunnynet.ops import CenteredScaleNorm, softwedge  # noqa: E402


def bench(fn, reps=13, warm=3):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return statistics.median(ts) * 1000


# ---------------- 候选: CSN 实现 ----------------

def csn_var_mean(self, x):
    """候选 B: var_mean 一次返回均值与方差."""
    var, mu = torch.var_mean(x, dim=-1, unbiased=False, keepdim=True)
    inv = torch.rsqrt(var + self.s_min * self.s_min + self.eps)
    return self.gamma * (x - mu + self.center) * inv


def csn_fused_sub(self, x):
    """候选 C: 先算 (x - mu + c), 再乘 inv; 用 add(alpha) 减少一次乘法."""
    mu = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, unbiased=False, keepdim=True)
    inv = torch.rsqrt(var + self.s_min * self.s_min + self.eps)
    return (x - mu + self.center) * (self.gamma * inv)


def csn_layer_norm_style(self, x):
    """候选 D: 用 F.layer_norm 的融合核 (weight=1,bias=0) 再做可学缩放, 数值近似."""
    y = F.layer_norm(x, (x.shape[-1],))
    return self.gamma * y + self.center * torch.rsqrt(
        x.var(dim=-1, unbiased=False, keepdim=True) + self.s_min ** 2 + self.eps)


def csn_fused_native(self, x):
    """候选 D2: native layer_norm 走 oneDNN 融合归约, 中心/增益后置."""
    d = x.shape[-1]
    y = F.layer_norm(x, (d,), eps=self.s_min * self.s_min + self.eps)
    return y * self.gamma + self.center


# ---------------- 候选: softwedge ----------------

def softwedge_simple(z, a):
    """候选 E: z + a*(0.5 + 0.5*z/(1+|z|)) 展开为 z + a/2 + (a/2)*z/(1+|z|)."""
    return z + a * 0.5 + (a * 0.5) * (z / (1.0 + z.abs()))


# ---------------- 候选: _emit 合并 ----------------

def make_emit_merged(orig):
    def emit(self, u, states, s_in):
        y = self._readout(u, None)
        sn = self.norm_s(states) if self.norm_s is not None else states
        fb = self._feedback(states, s_in)
        # 一次 matmul 合并 W_c(y) 与 W_s(fb)
        yy = torch.cat([y, fb], dim=-1)
        w = self._W_cat
        core = F.linear(yy, w) + self.b_gain * sn
        if self.use_state_mul:
            core = core + self.W_g(y) * sn
        return core
    return emit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    vocab, B, T = 8192, args.batch, args.seq_len
    torch.manual_seed(0)
    m = SunnyNet(SunnyConfig.small(vocab)).train()
    blk = m.blocks[0]
    D, I = blk.dim, blk.inner

    x = torch.randn(B, T, D, requires_grad=True)
    xin = torch.randn(B, T, D)
    s = torch.zeros(B, D)
    hist = torch.zeros(B, blk.taps - 1, D)

    def fwd_bwd(fn):
        def run():
            for p in m.parameters():
                p.grad = None
            out = fn()
            out.sum().backward()
            xin.grad = None
        return run

    results = {}

    # 基线
    def base_full():
        return blk(xin, s, hist)[0]

    results["base"] = bench(fwd_bwd(base_full))

    # A: 删 clamp (微基准直接测 clamp 在 (B,T,2D) 上的成本)
    t2d = torch.randn(B, T, 2 * D)
    results["clamp(2D) 单算子"] = bench(lambda: t2d.clamp(1e-3, 1.0))

    # CSN 候选
    for name, fn in (("csn:var_mean", csn_var_mean), ("csn:fused_sub", csn_fused_sub),
                     ("csn:native_ln", csn_fused_native)):
        orig = CenteredScaleNorm.forward
        CenteredScaleNorm.forward = fn
        results[name] = bench(fwd_bwd(base_full))
        CenteredScaleNorm.forward = orig

    # softwedge 候选
    import sunnynet.layers as L
    orig_sw = L.softwedge
    L.softwedge = softwedge_simple
    results["softwedge:simplified"] = bench(fwd_bwd(base_full))
    L.softwedge = orig_sw

    # _feedback 免 cat
    orig_fb = type(blk)._feedback

    def fb_nocat(self, states, s_in):
        B_, T_, D_ = states.shape
        prev = states.new_empty(B_, T_, D_)
        prev[:, 0] = s_in
        prev[:, 1:] = states[:, :-1]
        if self.circ is None:
            return prev
        return self.circ(prev)

    type(blk)._feedback = fb_nocat
    results["feedback:nocat"] = bench(fwd_bwd(base_full))
    type(blk)._feedback = orig_fb

    # 归约 microbench (直接对比 CSN 归约 vs native LN)
    xn = torch.randn(B * T, D)
    results["micro: mean+var"] = bench(lambda: (xn.mean(-1), xn.var(-1, unbiased=False)))
    results["micro: var_mean"] = bench(lambda: torch.var_mean(xn, dim=-1, unbiased=False))
    results["micro: layer_norm"] = bench(lambda: F.layer_norm(xn, (D,)))

    # 排序输出
    print(f"\n单块前向+反向 (B={B} T={T} threads={args.threads})  dim={D} inner={I}")
    b0 = results["base"]
    for k, v in sorted(results.items(), key=lambda kv: kv[1]):
        print(f"  {k:26s}{v:9.3f} ms   {v/b0*100:6.1f}% of base")


if __name__ == "__main__":
    main()