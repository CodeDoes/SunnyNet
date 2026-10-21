"""GGUF 驱动的 RWKV-7 与官方 .pth 参考实现的逐 token 数值等价.

参考数学: BlinkDL/RWKV-LM 的 RWKV-v7/rwkv_v7_numpy.py (johanwind 纯 numpy 参考).
本脚本用 pth 权重独立实现一遍, 与 sunnynet/runtime/rwkv7.py (GGUF 驱动) 比对.
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from sunnynet.runtime.rwkv7 import RWKV7  # noqa: E402

GGUF = ROOT / "experiments/gguf/rwkv7-191M-world-bf16.gguf"
PTH = ROOT / "experiments/gguf/rwkv-x070-0.1b-v2.8.pth"
OUT = ROOT / "experiments/results/_rwkv7_gguf_equiv.json"


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


class Ref:
    """按官方 numpy 参考实现的 .pth 版本."""

    def __init__(self, pth):
        w = torch.load(pth, map_location="cpu", weights_only=True)
        self.w = {k: v.float().numpy() for k, v in w.items()}
        self.L = 12
        self.D = 768
        self.HS = 64
        self.NH = self.D // self.HS
        self.eps = 1e-5

    def _t(self, k):
        a = self.w[k]
        return a.reshape(-1) if a.ndim > 2 else a

    def init(self):
        return {"S": np.zeros((self.L, self.NH, self.HS, self.HS), dtype=np.float32),
                "xf": np.zeros((self.L, 2, self.D), dtype=np.float32)}

    def step(self, tok, st):
        w, L, D, HS, NH, eps = self.w, self.L, self.D, self.HS, self.NH, self.eps
        S_all, xf = st["S"], st["xf"]
        x = w["emb.weight"][tok].copy()
        x = (x - x.mean()) / np.sqrt(x.var() + eps) * w["blocks.0.ln0.weight"] + w["blocks.0.ln0.bias"]

        v0 = None
        for i in range(L):
            p = f"blocks.{i}."
            xn = (x - x.mean()) / np.sqrt(x.var() + eps) * w[p + "ln1.weight"] + w[p + "ln1.bias"]
            last = xf[i, 0]
            d = last - xn
            xr = xn + self._t(p + "att.x_r") * d
            xw = xn + self._t(p + "att.x_w") * d
            xk = xn + self._t(p + "att.x_k") * d
            xv = xn + self._t(p + "att.x_v") * d
            xa = xn + self._t(p + "att.x_a") * d
            xg = xn + self._t(p + "att.x_g") * d

            r = w[p + "att.receptance.weight"] @ xr
            ww = np.exp(-sig(np.tanh(xw @ w[p + "att.w1"]) @ w[p + "att.w2"]
                             + self._t(p + "att.w0")) / np.sqrt(np.e))
            k = w[p + "att.key.weight"] @ xk
            v = w[p + "att.value.weight"] @ xv
            if v0 is None:
                v0 = v
            else:
                v = v + (v0 - v) * sig(xv @ w[p + "att.v1"] @ w[p + "att.v2"]
                                       + self._t(p + "att.v0"))
            a = sig(xa @ w[p + "att.a1"] @ w[p + "att.a2"] + self._t(p + "att.a0"))
            g = sig(xg @ w[p + "att.g1"]) @ w[p + "att.g2"]

            kk = k * self._t(p + "att.k_k")
            k = k + k * (a - 1.0) * self._t(p + "att.k_a")

            r = r.reshape(NH, HS); ww = ww.reshape(NH, HS)
            k = k.reshape(NH, HS); v = v.reshape(NH, HS); a = a.reshape(NH, HS)
            kk = kk.reshape(NH, HS)
            kk = kk / np.maximum(np.linalg.norm(kk, axis=1, keepdims=True), 1e-12)

            S = S_all[i]
            S = S * ww[:, None, :] - np.einsum("hij,hj->hi", S, kk)[:, :, None] * (kk * a)[:, None, :] \
                + v[:, :, None] * k[:, None, :]
            S_all[i] = S
            y = np.einsum("hij,hj->hi", S, r)
            m = y.mean(axis=1, keepdims=True); var = y.var(axis=1, keepdims=True)
            y = ((y - m) / np.sqrt(var + 64e-5)).reshape(-1) * self._t(p + "att.ln_x.weight") + self._t(p + "att.ln_x.bias")
            rk = self.w[p + "att.r_k"].reshape(NH, HS)
            y = y + (((r * k * rk).sum(axis=1, keepdims=True) * v).reshape(-1))
            y = y * g
            x = x + w[p + "att.output.weight"] @ y
            xf[i, 0] = xn

            xn2 = (x - x.mean()) / np.sqrt(x.var() + eps) * w[p + "ln2.weight"] + w[p + "ln2.bias"]
            last2 = xf[i, 1]
            ck = np.maximum(w[p + "ffn.key.weight"] @ (xn2 + self._t(p + "ffn.x_k") * (last2 - xn2)), 0.0)
            x = x + w[p + "ffn.value.weight"] @ (ck * ck)
            xf[i, 1] = xn2

        x = (x - x.mean()) / np.sqrt(x.var() + eps) * w["ln_out.weight"] + w["ln_out.bias"]
        return w["head.weight"] @ x


def main():
    ref = Ref(PTH)
    mine = RWKV7(GGUF)

    text = "\nIn a shocking finding, scientist discovered a herd of dragons"
    ids = mine.tok.encode(text)
    print(f"prompt tokens: {len(ids)}")

    rs = ref.init()
    ms = mine.reset(mine.init_state())
    rows = []
    for i, t in enumerate(ids[:12]):
        rl = ref.step(t, rs)
        ml = mine.step(t, ms)
        d = float(np.abs(rl - ml).max())
        am_r, am_m = int(np.argmax(rl)), int(np.argmax(ml))
        rows.append({"step": i, "max_abs_diff": d,
                     "argmax_ref": am_r, "argmax_mine": am_m, "agree": am_r == am_m})
        print(f"  step {i:2d}: max|diff| {d:.3e}  argmax {am_r}/{am_m} {'OK' if am_r==am_m else 'MISMATCH'}")

    agree = sum(1 for r in rows if r["agree"])
    summary = {
        "prompt_tokens": len(ids),
        "steps": len(rows),
        "max_abs_diff": max(r["max_abs_diff"] for r in rows),
        "argmax_agree": f"{agree}/{len(rows)}",
        "rows": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nmax|diff| overall {summary['max_abs_diff']:.3e}  argmax {summary['argmax_agree']}")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()