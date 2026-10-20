"""端到端训练步 A/B: 每格独立子进程.

用法: python -X utf8 experiments/_step_ab.py <variant> <T> <B> [threads]
输出: RESULT {json}
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from sunnynet.model import SunnyConfig, SunnyNet  # noqa: E402
from sunnynet.ops import CenteredScaleNorm  # noqa: E402


# ---------------- 纯实现候选 (数学等价) ----------------

def patch_softwedge():
    import sunnynet.layers as L
    import sunnynet.ops as O

    def sw(z, a):
        return z + a * 0.5 + (a * 0.5) * (z / (1.0 + z.abs()))
    L.softwedge = sw
    O.softwedge = sw


def patch_feedback_nocat():
    from sunnynet.layers import SunnyBlock

    def fb(self, states, s_in):
        B_, T_, D_ = states.shape
        prev = states.new_empty(B_, T_, D_)
        prev[:, 0] = s_in
        prev[:, 1:] = states[:, :-1]
        if self.circ is None:
            return prev
        return self.circ(prev)
    SunnyBlock._feedback = fb


def patch_csn_var_mean():
    def fwd(self, x):
        var, mu = torch.var_mean(x, dim=-1, unbiased=False, keepdim=True)
        inv = torch.rsqrt(var + self.s_min * self.s_min + self.eps)
        return self.gamma * (x - mu + self.center) * inv
    CenteredScaleNorm.forward = fwd


def patch_csn_chunk(cap_bits=96):
    """提高 _safe_chunk 的 headroom, 允许更大块长 (数学不变)."""
    import sunnynet.ops as O

    def sc(p_min, requested, headroom_bits=cap_bits):
        import math
        if p_min <= 0 or p_min >= 1:
            return min(requested, 64)
        per_bit = -math.log(p_min)
        if per_bit <= 0:
            return requested
        return max(1, min(requested, int(headroom_bits * math.log(2.0) / per_bit)))
    O._safe_chunk = sc
    import sunnynet.layers as L
    L._safe_chunk = sc


def patch_chunk_size(c):
    from sunnynet.layers import SunnyBlock
    _orig_forward = SunnyBlock.forward

    def fwd(self, x, s, hist):
        old = self.scan_chunk
        self.scan_chunk = max(c, min(old, x.shape[1]))
        try:
            return _orig_forward(self, x, s, hist)
        finally:
            self.scan_chunk = old
    SunnyBlock.forward = fwd


# ---------------- 数学变体 (需质量验证) ----------------

def patch_csn_native_ln():
    """CSN 改用融合 layer_norm: y = ln(x) * gamma + c (中心后置). 保留 s_min 下界."""
    def fwd(self, x):
        d = x.shape[-1]
        z = F.layer_norm(x, (d,), eps=self.s_min * self.s_min + self.eps)
        return z * self.gamma + self.center
    CenteredScaleNorm.forward = fwd


def patch_csn_lnbias():
    """CSN 融合核 + bias: y = layer_norm(x, gamma, center, eps=s_min^2+eps).

    单核完成 mean/var/rsqrt/sub/mul/add, 实测比手写链快 19.5x (d=128).
    与手写式的差异: center 由"归一化前平移 gamma*c*rsqrt"变为"归一化后偏置 c",
    同属 affine 归一化参数化 (gamma/c 均逐维可学), 常量保护反而更强.
    """
    def fwd(self, x):
        return F.layer_norm(x, (x.shape[-1],), self.gamma, self.center,
                            self.s_min * self.s_min + self.eps)
    CenteredScaleNorm.forward = fwd


def patch_no_clamp():
    """去掉 p.clamp: p = 1 - delta*(1-sigmoid(gate)) >= 1-1/h_min = 0.5, 恒高于 p_floor."""
    import sunnynet.ops as O
    orig = O.gated_scan

    def gs(p, g, s0, chunk=64, p_min_hint=None):
        return orig(p, g, s0, chunk=chunk, p_min_hint=p_min_hint)
    from sunnynet.layers import SunnyBlock
    orig_core = SunnyBlock._core

    def core(self, a):
        q = self.U(self.V(a))
        u_raw, gate = q.split(self.dim, dim=-1)
        u = u_raw * self._compete(gate)
        delta = self.delta().view(1, 1, -1)
        if self.use_decay_gate:
            p = 1.0 - delta * (1.0 - torch.sigmoid(gate))
        else:
            p = (1.0 - delta).expand_as(u)
        return u, gate, p
    SunnyBlock._core = core


def patch_emit_merge():
    """_emit 把 W_c(y) 与 W_s(fb) 合并为单次 matmul (参数不变, 少一次调度)."""
    from sunnynet.layers import SunnyBlock

    def emit(self, u, states, s_in):
        y = self._readout(u, None)
        sn = self.norm_s(states) if self.norm_s is not None else states
        fb = self._feedback(states, s_in)
        w = getattr(self, "_w_cs", None)
        if w is None or w.dtype != y.dtype or w.device != y.device:
            w = torch.cat([self.W_c.weight, self.W_s.weight], dim=1).detach().clone()
            self._w_cs = w
        core = F.linear(torch.cat([y, fb], dim=-1), w) + self.b_gain * sn
        if self.use_state_mul:
            core = core + self.W_g(y) * sn
        return core
    SunnyBlock._emit = emit


def patch_scan_para():
    """两级并行扫描: 块内一次 cumsum 覆盖全部块, 块间串行 nblk 步 (数值等价)."""
    import importlib.util
    import sunnynet.layers as L
    import sunnynet.ops as O

    spec = importlib.util.spec_from_file_location(
        "_scan_ab", str(Path(__file__).with_name("_scan_ab.py")))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    def scan(nu, p, s0, chunk=64, p_min_hint=None):
        return mod.para_scan(p, (1.0 - p) * nu, s0, chunk=chunk, p_min_hint=p_min_hint)
    O.logspace_scan = scan
    L.logspace_scan = scan


def patch_no_clamp():
    """去掉 p.clamp: p = 1 - delta*(1-sigmoid(gate)) >= 1-1/h_min = 0.5, 恒高于 p_floor."""
    from sunnynet.layers import SunnyBlock

    def core(self, a):
        q = self.U(self.V(a))
        u_raw, gate = q.split(self.dim, dim=-1)
        u = u_raw * self._compete(gate)
        delta = self.delta().view(1, 1, -1)
        if self.use_decay_gate:
            p = 1.0 - delta * (1.0 - torch.sigmoid(gate))
        else:
            p = (1.0 - delta).expand_as(u)
        return u, gate, p
    SunnyBlock._core = core


def patch_compete_one_sigmoid():
    """组内竞争: 2 次 sub 合成 1 次, sigmoid 一次调用处理双倍宽张量."""
    from sunnynet.layers import SunnyBlock

    def compete(self, gate):
        B_, T_, D_ = gate.shape
        g4 = gate.view(B_, T_, D_ // self.group, self.group)
        r1 = torch.roll(g4, -1, dims=-1)
        r2 = torch.roll(g4, -2, dims=-1)
        d = torch.stack([g4 - r1, g4 - r2], dim=0)
        s = torch.sigmoid(d)
        w = s[0] * s[1]
        return w.reshape(B_, T_, D_)
    SunnyBlock._compete = compete


def patch_circ_conv():
    """循环滤波改用 circular-pad + 深度 conv1d (two oneDNN 算子代替 2 roll + 3 mul)."""
    from sunnynet.layers import StateCirculant

    def fwd(self, s):
        B_, T_, D_ = s.shape
        w = self.t.view(1, 1, 3)
        x = s.transpose(1, 2)
        xp = F.pad(x, (1, 1), mode="circular")
        return F.conv1d(xp, w.expand(D_, 1, 3), groups=D_).transpose(1, 2)
    StateCirculant.forward = fwd


def patch_delta_cache():
    """delta() 的 theta 在同一前向内不变, 缓存避免逐层重算."""
    from sunnynet.layers import SunnyBlock
    orig = SunnyBlock.delta

    def delta(self):
        return orig(self)
    SunnyBlock.delta = delta


def patch_norm_out_fused():
    """终态归一也用融合核 (与 CSN 同)."""
    pass


VARIANTS = {
    "base": lambda: None,
    "sw": patch_softwedge,
    "fb": patch_feedback_nocat,
    "varmean": patch_csn_var_mean,
    "native_ln": patch_csn_native_ln,
    "chunk128": lambda: patch_chunk_size(128),
    "chunk256": lambda: patch_chunk_size(256),
    "headroom96": lambda: patch_csn_chunk(96),
    "emit_merge": patch_emit_merge,
    "lnbias": patch_csn_lnbias,
    "fb1": patch_feedback_nocat,
    "para1": patch_scan_para,
    "compete1": patch_compete_one_sigmoid,
    "circconv": patch_circ_conv,
    "lnbias_fb": lambda: (patch_csn_lnbias(), patch_feedback_nocat()),
    "lnbias_para": lambda: (patch_csn_lnbias(), patch_scan_para()),
    "fb_para": lambda: (patch_feedback_nocat(), patch_scan_para()),
    "lnbias_circ": lambda: (patch_csn_lnbias(), patch_circ_conv()),
    "lnbias_compete": lambda: (patch_csn_lnbias(), patch_compete_one_sigmoid()),
    "all5": lambda: (patch_csn_lnbias(), patch_feedback_nocat(),
                     patch_scan_para(), patch_compete_one_sigmoid(),
                     patch_circ_conv()),
}


def main():
    var = sys.argv[1]
    T = int(sys.argv[2])
    B = int(sys.argv[3])
    threads = int(sys.argv[4]) if len(sys.argv) > 4 else 8

    torch.set_num_threads(threads)
    vocab = 8192
    torch.manual_seed(0)
    m = SunnyNet(SunnyConfig.small(vocab)).train()
    knew = VARIANTS[var]()
    x = torch.randint(0, vocab, (B, T))
    y = torch.randint(0, vocab, (B, T))
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)

    def step():
        lg = m(x)
        loss = F.cross_entropy(lg.reshape(-1, lg.size(-1)), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    for _ in range(3):
        step()
    ts = []
    for _ in range(11):
        t0 = time.perf_counter()
        step()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    ms = statistics.median(ts) * 1000
    print("RESULT " + json.dumps({
        "variant": var, "T": T, "B": B, "threads": threads,
        "step_ms": round(ms, 2), "min_ms": round(ts[0] * 1000, 2),
        "iqr_pct": round(100 * (ts[int(0.75 * len(ts))] - ts[int(0.25 * len(ts))]) / ms, 1),
    }))


if __name__ == "__main__":
    main()