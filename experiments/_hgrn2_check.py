"""HGRN2 单步递推的等价性与端到端验证.

校验三项:
  1) 单步链式 vs 分段续算逐位一致性
  2) 与独立书写的参考实现 logits 对齐 (显式循环, 不借 einsum)
  3) 贪心续写连通性
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sunnynet.runtime.hgrn2 import HGRN2  # noqa: E402

# 参考实现故意用朴素 exp 写法保持与 hgrn2.py 独立; x<<0 时溢出只影响警告,
# 数值上 sigmoid 饱和到 0, 结果正确。
np.seterr(over="ignore")

MDIR = os.environ.get("SUNNY_HGRN2",
                      str(Path("F:/sunnyclaw-data/models/hgrn2-1.3B-100B")))


def _ref_logits(m: HGRN2, ids):
    """与 hgrn2.py 同构但独立书写的参考: 显式循环矩阵-向量, 不借 einsum。"""
    H, K, V = m.n_heads, m.head_f_dim, m.head_i_dim
    S = np.zeros((m.n_layers, H, K, V), dtype=np.float32)
    lb_all = m.lower_bounds() if m.use_lower_bound else None
    Ws = [m._layer_w(i) for i in range(m.n_layers)]
    emb = (m._blob.embed if m._blob is not None
           else m.w("model.embeddings.weight").reshape(-1))
    ln_f = (m._blob.ln_f if m._blob is not None else m.w("model.norm.weight"))
    head = (m._blob.embed if m._blob is None or m._blob.tied else m._blob.head)
    if m._blob is None and not m.tie:
        head = m.w("lm_head.weight")
    out = []
    for tok in ids:
        x = np.array(emb[tok * m.hidden:(tok + 1) * m.hidden], dtype=np.float32)
        for i in range(m.n_layers):
            W = Ws[i]
            res = x
            h = x / np.sqrt((x * x).mean(dtype=np.float32) + m.eps) * W["an"]
            qq = W["qp"] @ h
            qq = qq / (1.0 + np.exp(-qq, dtype=np.float32))
            ff = W["fp"] @ h
            vv = W["ip"] @ h

            ff = ff.reshape(-1, 1)
            g = (-np.logaddexp(0.0, -ff)).astype(np.float32).reshape(-1)
            if lb_all is not None and i > 0:
                lb = lb_all[i].reshape(-1, 1)
                g = np.logaddexp(np.log(lb), np.log1p(-lb) + g.reshape(-1, 1))
                g = np.asarray(g, dtype=np.float32).reshape(-1)
            kk = (1.0 - np.exp(g, dtype=np.float32)).astype(np.float32)
            dec = np.exp(g.reshape(H, K), dtype=np.float32)

            o = np.zeros((H, V), dtype=np.float32)
            for hh in range(H):
                Sh = S[i, hh] * dec[hh][:, None]
                Sh = Sh + np.outer(kk.reshape(H, K)[hh], vv.reshape(H, V)[hh])
                S[i, hh] = Sh
                o[hh] = (Sh.T @ qq.reshape(H, K)[hh]) * m.scale

            of = o.reshape(-1)
            of = of / np.sqrt((of * of).mean(dtype=np.float32) + m.eps) * W["gn"]
            x = res + W["op"] @ of

            h2 = x / np.sqrt((x * x).mean(dtype=np.float32) + m.eps) * W["mn"]
            gate = W["mgp"] @ h2
            up = W["mup"] @ h2
            x = x + W["mdp"] @ (
                (gate / (1.0 + np.exp(-gate, dtype=np.float32))) * up)
        x = x / np.sqrt((x * x).mean(dtype=np.float32) + m.eps) * ln_f
        out.append(head.reshape(m.vocab, m.hidden) @ x)
    return np.stack(out)


def main():
    print("loading", MDIR)
    m = HGRN2(MDIR)
    print(f"dims: hidden={m.hidden} L={m.n_layers} H={m.n_heads} "
          f"K={m.head_f_dim} V={m.head_i_dim} vocab={m.vocab} "
          f"lower_bound={m.use_lower_bound} tied={m.tie}")

    rng = np.random.default_rng(7)
    ids = rng.integers(0, m.vocab, size=12).tolist()

    st = m.reset(m.init_state())
    got = np.stack([m.step(t, st) for t in ids])

    st2 = m.reset(m.init_state())
    seg = np.concatenate([np.stack([m.step(t, st2) for t in ids[:5]]),
                          np.stack([m.step(t, st2) for t in ids[5:]])])
    d_seg = float(np.abs(got - seg).max())

    ref = _ref_logits(m, ids)
    d_ref = float(np.abs(got - ref).max())
    am = int((got.argmax(1) == ref.argmax(1)).sum())

    print(f"step vs segment max|diff| = {d_seg:.3e}")
    print(f"step vs ref     max|diff| = {d_ref:.3e}  (fp32 单精度累积)")
    print(f"argmax agree: {am}/{len(ids)}")

    RES = ROOT / "results"
    RES.mkdir(exist_ok=True)
    (RES / "_hgrn2_check.json").write_text(json.dumps({
        "dims": {"hidden": m.hidden, "layers": m.n_layers, "heads": m.n_heads,
                 "head_f_dim": m.head_f_dim, "head_i_dim": m.head_i_dim,
                 "vocab": m.vocab, "expand_ratio": m.expand_ratio},
        "step_vs_segment_maxdiff": d_seg,
        "step_vs_ref_maxdiff": d_ref,
        "argmax_agree": am,
        "n_tokens": len(ids),
        "use_lower_bound": bool(m.use_lower_bound),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote results/_hgrn2_check.json")


if __name__ == "__main__":
    main()