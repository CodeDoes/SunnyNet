"""GLA 单步递推的等价性与端到端验证.

参考实现独立写在 experiments/_gla_ref.py, 与本仓 gla.py 不共享代码,
避免"自己验自己"。校验三项:
  1) 单步链式 vs 整段 numpy 参考逐位一致性 (相同实现路径的重复性)
  2) 与独立参考实现的 logits 对齐
  3) 贪心续写的连通性
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sunnynet.runtime.gla import GLA  # noqa: E402

MDIR = os.environ.get("SUNNY_GLA",
                      str(Path("E:/sunnyclaw-data/models/gla-340M")))


def _ref_logits(m: GLA, ids):
    """与 gla.py 同构但独立书写的参考: 显式循环矩阵-向量, 不借 einsum."""
    H, K, V = m.n_heads, m.head_k_dim, m.head_v_dim
    S = np.zeros((m.n_layers, H, K, V), dtype=np.float32)
    out = []
    for tok in ids:
        x = m.w("model.embeddings.weight")[tok].astype(np.float32)
        for i in range(m.n_layers):
            p = f"model.layers.{i}."
            res = x
            h = x / np.sqrt((x * x).mean(dtype=np.float32) + m.eps) * m.w(p + "attn_norm.weight")
            q = m.w(p + "attn.q_proj.weight") @ h
            k = m.w(p + "attn.k_proj.weight") @ h
            v = m.w(p + "attn.v_proj.weight") @ h
            gk = (m.w(p + "attn.gk_proj.1.weight")
                  @ (m.w(p + "attn.gk_proj.0.weight") @ h)
                  + m.w(p + "attn.gk_proj.1.bias"))
            gk = -np.logaddexp(0.0, -gk).astype(np.float32) / m.gate_logit_normalizer
            dec = np.exp(gk.reshape(H, K), dtype=np.float32)
            o = np.zeros((H, V), dtype=np.float32)
            for hh in range(H):
                Sh = S[i, hh] * dec[hh][:, None]
                Sh = Sh + np.outer(k.reshape(H, K)[hh], v.reshape(H, V)[hh])
                S[i, hh] = Sh
                o[hh] = Sh.T @ q.reshape(H, K)[hh] * m.scale
            g = m.w(p + "attn.g_proj.weight") @ h
            o = o / np.sqrt((o * o).mean(axis=-1, keepdims=True, dtype=np.float32) + m.eps) \
                * m.w(p + "attn.g_norm_swish_gate.weight")
            o = (o * (g.reshape(H, V) / (1.0 + np.exp(-g.reshape(H, V),
                                                       dtype=np.float32)))).reshape(-1)
            x = res + m.w(p + "attn.o_proj.weight") @ o
            h2 = x / np.sqrt((x * x).mean(dtype=np.float32) + m.eps) * m.w(p + "mlp_norm.weight")
            gate = m.w(p + "mlp.gate_proj.weight") @ h2
            up = m.w(p + "mlp.up_proj.weight") @ h2
            x = x + m.w(p + "mlp.down_proj.weight") @ (
                (gate / (1.0 + np.exp(-gate, dtype=np.float32))) * up)
        x = x / np.sqrt((x * x).mean(dtype=np.float32) + m.eps) * m.w("model.norm.weight")
        out.append(m.w("model.embeddings.weight") @ x)
    return np.stack(out)


def main():
    print("loading", MDIR)
    m = GLA(MDIR)
    print(f"dims: hidden={m.hidden} L={m.n_layers} H={m.n_heads} "
          f"K={m.head_k_dim} V={m.head_v_dim} vocab={m.vocab} "
          f"gate_rank={m.gate_rank} normalizer={m.gate_logit_normalizer}")

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

    print(f"step vs segment max|diff| = {d_seg:.3e}")
    print(f"step vs ref     max|diff| = {d_ref:.3e}  (fp32 单精度累积)")
    print(f"argmax agree: {(got.argmax(1) == ref.argmax(1)).sum()}/{len(ids)}")

    # 贪心续写 (config 的 add_bos_token=true, 手工 prepend <s>=1)
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.Load(str(Path(MDIR) / "tokenizer.model"))
    prompt = [1] + sp.encode("The capital of France is", out_type=int)
    gen = m.generate(prompt, max_new=20, temperature=0.0)
    text = sp.decode(gen)
    print("greedy:", repr(text))

    RES = ROOT / "results"
    RES.mkdir(exist_ok=True)
    import json
    (RES / "_gla_check.json").write_text(json.dumps({
        "dims": {"hidden": m.hidden, "layers": m.n_layers, "heads": m.n_heads,
                 "head_k_dim": m.head_k_dim, "head_v_dim": m.head_v_dim,
                 "vocab": m.vocab},
        "step_vs_segment_maxdiff": d_seg,
        "step_vs_ref_maxdiff": d_ref,
        "argmax_agree": int((got.argmax(1) == ref.argmax(1)).sum()),
        "n_tokens": len(ids),
        "greedy": text,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote results/_gla_check.json")


if __name__ == "__main__":
    main()