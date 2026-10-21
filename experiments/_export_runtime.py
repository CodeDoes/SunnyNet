"""导出 SunnyNet 权重为 C Runtime 的打包 fp32 布局.

布局 (整块 fp32, 前置 10 个 int32 头):
  header: magic, vocab, dim, inner, rank, taps, layers, group, tied, clamp
  per layer (顺序固定, 见 sn_load):
    norm_in.gamma (D)   norm_in.center (D)
    norm_mix.gamma (I)  norm_mix.center (I)
    dw.weight^T   (K,D)         需按 k 维重排
    V.weight^T    (D,R)
    U.weight^T    (R,2D)
    W_z.weight^T  (D,I)
    W_c.weight^T  (I,D)
    W_s_eff^T     (D,D)         已折入 circ
    delta (D)
    a_tanh, b_gain (2)
  尾部: eps, p_floor, norm_out.gamma (D), norm_out.center (D),
        embed (vocab,D), head^T (D,vocab)
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sunnynet.cli import load_model  # noqa: E402

MAGIC = 0x4E4E5355
ALLOWED = dict(decay_gate=True, circulant=True, local_taps=True, current=True,
               residual=True, state_norm=False, state_mul=False)


def ws_eff(blk) -> torch.Tensor:
    W = blk.W_s.weight
    t0, t1, t2 = blk.circ.t[0], blk.circ.t[1], blk.circ.t[2]
    return t0 * torch.roll(W, -1, dims=1) + t1 * W + t2 * torch.roll(W, 1, dims=1)


def export(ckpt: str, out: str) -> dict:
    m, cfg, _ = load_model(ckpt)
    m.eval()
    ab = cfg.ablation or {}
    for k, want in ALLOWED.items():
        got = ab.get(k, want)
        if got != want:
            raise SystemExit(f"该 ckpt 的 ablation[{k}]={got}, Runtime 只支持默认路径")

    D, I, R, K, L = cfg.dim, cfg.inner, cfg.rank, cfg.taps, cfg.layers
    if not cfg.tie_embeddings:
        raise SystemExit("Runtime 只支持绑定嵌入 (tie_embeddings=True)")
    head_tied = cfg.tie_embeddings
    p_floor = 1.0 / cfg.h_max
    # CenteredScaleNorm 的分母是 var + s_min^2 + eps, s_min^2 比 eps 大两个量级.
    eps = 0.0
    needs_clamp = any(b._need_clamp for b in m.blocks)
    for b in m.blocks:
        nm = b.norm_in
        eps = max(eps, nm.s_min * nm.s_min + nm.eps)

    parts = []
    hdr = struct.pack("<10i", MAGIC, cfg.vocab_size, D, I, R, K, L, 8,
                      int(head_tied), int(needs_clamp))
    parts.append(hdr)

    def put(t):
        parts.append(t.detach().float().contiguous().view(-1).numpy().tobytes())

    for b in m.blocks:
        put(b.norm_in.gamma)
        put(b.norm_in.center)
        put(b.norm_mix.gamma)
        put(b.norm_mix.center)
        # dw.weight (D,1,K) -> (K,D)
        put(b.dw.weight.detach().reshape(D, K).transpose(0, 1).contiguous())
        put(b.V.weight.transpose(0, 1).contiguous())       # (D,R)
        put(b.U.weight.transpose(0, 1).contiguous())       # (R,2D)
        put(b.W_z.weight.transpose(0, 1).contiguous())     # (D,I)
        put(b.W_c.weight.transpose(0, 1).contiguous())     # (I,D)
        put(ws_eff(b).transpose(0, 1).contiguous())        # (D,D)
        put(b.delta().detach())
        parts.append(struct.pack("<2f", float(torch.tanh(b.a.detach())),
                                 float(b.b_gain.detach())))

    parts.append(struct.pack("<2f", eps, p_floor))
    put(m.norm_out.gamma)
    put(m.norm_out.center)
    put(m.embed.weight)                                    # (vocab,D)
    put(m.embed.weight.transpose(0, 1).contiguous())       # (D,vocab)

    buf = b"".join(parts)
    Path(out).write_bytes(buf)
    n_params = m.num_params()
    return {"file": out, "bytes": len(buf), "params": n_params,
            "dim": D, "inner": I, "rank": R, "taps": K, "layers": L,
            "vocab": cfg.vocab_size, "needs_clamp": needs_clamp,
            "bytes_per_param": len(buf) / max(n_params, 1)}


if __name__ == "__main__":
    ck = sys.argv[1] if len(sys.argv) > 1 else "experiments/ckpts_ood/sunnynet.pt"
    o = sys.argv[2] if len(sys.argv) > 2 else "experiments/ckpts_ood/sunnynet.runtime.bin"
    import json
    print(json.dumps(export(ck, o), ensure_ascii=False, indent=2))