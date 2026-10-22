"""DeltaNet 单步递推, 权重从 HF safetensors 读入.

数学依据: fla/layers/delta_net.py 的 fused_recurrent 路径
与 fla/ops/delta_rule/fused_recurrent.py 的单步形式。

状态是逐 head 的 (k_dim, v_dim) 矩阵, delta rule 含 rank-1 校正:
    delta_t = (v_t - S_{t-1}^T k_t) * beta_t
    S_t     = S_{t-1} + k_t ⊗ delta_t
    o_t     = S_t^T q_t
这是矩阵值递推, 元素级 SN_FMA 无法表达, 需要专门的 rank-1 内核。

维度 (config.json): hidden 1024, num_heads 8, head_k_dim=head_v_dim=128,
conv_size 4, headwise beta (b_proj 输出 8), qk_norm l2 + silu, qk_activation silu,
o_norm 为无门控 RMSNorm(128), use_gate=false ⇒ 无衰减门。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _silu(x):
    return x / (1.0 + np.exp(-x, dtype=np.float32))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x, dtype=np.float32))


def _rms_norm(x, w, eps):
    return x / np.sqrt((x * x).mean(axis=-1, keepdims=True, dtype=np.float32) + eps) * w


def _l2norm(x, eps=1e-6):
    n = np.sqrt((x * x).sum(axis=-1, keepdims=True, dtype=np.float32) + eps)
    return x / n


class DeltaNet:
    """delta rule 单步推理; 每步 O(1) 与上下文长度无关."""

    def __init__(self, model_dir: str | Path):
        self.dir = Path(model_dir)
        self._load()
        self._infer_dims()

    # ------------------------------------------------------------------ 加载
    def _load(self):
        import torch
        from safetensors.torch import load_file

        p = self.dir / "model.safetensors"
        sd = load_file(str(p)) if p.exists() else torch.load(
            self.dir / "pytorch_model.bin", map_location="cpu", weights_only=True)
        self._np = {k: v.detach().to(torch.float32).numpy() for k, v in sd.items()}

    def w(self, name: str) -> np.ndarray:
        return self._np[name]

    def _infer_dims(self):
        self.hidden = self.w("model.embeddings.weight").shape[1]
        self.vocab = self.w("model.embeddings.weight").shape[0]
        self.n_layers = len({k.split(".")[2] for k in self._np
                             if k.startswith("model.layers.")})
        self.n_heads = self.w("model.layers.0.attn.b_proj.weight").shape[0]
        self.head_k_dim = self.w("model.layers.0.attn.q_proj.weight").shape[0] // self.n_heads
        self.head_v_dim = self.w("model.layers.0.attn.v_proj.weight").shape[0] // self.n_heads
        self.key_dim = self.head_k_dim * self.n_heads
        self.value_dim = self.head_v_dim * self.n_heads
        self.conv_size = self.w("model.layers.0.attn.q_conv1d.weight").shape[-1]
        self.eps = 1e-6
        self.scale = self.head_k_dim ** -0.5
        self.headwise_beta = self.w("model.layers.0.attn.b_proj.weight").shape[0] == self.n_heads

    # ------------------------------------------------------------------ 状态
    def init_state(self):
        L, H, K, V = self.n_layers, self.n_heads, self.head_k_dim, self.head_v_dim
        return {
            # delta rule 状态: (heads, k_dim, v_dim)
            "S": np.zeros((L, H, K, V), dtype=np.float32),
            # 三路短卷积各存 (conv_size-1) 步历史
            "cq": np.zeros((L, self.conv_size - 1, self.key_dim), dtype=np.float32),
            "ck": np.zeros((L, self.conv_size - 1, self.key_dim), dtype=np.float32),
            "cv": np.zeros((L, self.conv_size - 1, self.value_dim), dtype=np.float32),
        }

    def reset(self, state):
        for v in state.values():
            v[...] = 0.0
        return state

    # ------------------------------------------------------------- 短卷积
    def _conv(self, x, hist, kw):
        """深度因果卷积 + silu; hist 原地滑动."""
        k = self.conv_size
        acc = kw[:, -1] * x
        for t in range(k - 1):
            acc += kw[:, t] * hist[t]
        if k > 1:
            hist[:-1] = hist[1:]
            hist[-1] = x
        return _silu(acc)

    # ------------------------------------------------------------------ 前向
    def step(self, token: int, state) -> np.ndarray:
        H, K, V = self.n_heads, self.head_k_dim, self.head_v_dim
        S_all = state["S"]

        x = self.w("model.embeddings.weight")[token].astype(np.float32)

        for i in range(self.n_layers):
            p = f"model.layers.{i}."
            res = x
            h = _rms_norm(x, self.w(p + "attn_norm.weight"), self.eps)

            # 三路投影各自过深度短卷积 (kernel=4, 无偏置), 再 silu
            qr = self.w(p + "attn.q_proj.weight") @ h
            kr = self.w(p + "attn.k_proj.weight") @ h
            vr = self.w(p + "attn.v_proj.weight") @ h
            q = self._conv(qr, state["cq"][i],
                           self.w(p + "attn.q_conv1d.weight").reshape(self.key_dim, self.conv_size))
            k = self._conv(kr, state["ck"][i],
                           self.w(p + "attn.k_conv1d.weight").reshape(self.key_dim, self.conv_size))
            v = self._conv(vr, state["cv"][i],
                           self.w(p + "attn.v_conv1d.weight").reshape(self.value_dim, self.conv_size))

            q2 = _l2norm(q.reshape(H, K)) * self.scale
            k2 = _l2norm(k.reshape(H, K))
            v2 = v.reshape(H, V)

            beta = _sigmoid(self.w(p + "attn.b_proj.weight") @ h)
            if not self.headwise_beta:
                beta = np.full(H, float(beta), dtype=np.float32)

            S = S_all[i]
            kv = np.einsum("hkv,hk->hv", S, k2)
            delta = (v2 - kv) * beta[:, None]
            S = S + k2[:, :, None] * delta[:, None, :]
            S_all[i] = S

            o = np.einsum("hkv,hk->hv", S, q2)
            o = _rms_norm(o, self.w(p + "attn.o_norm.weight"), self.eps).reshape(-1)
            x = res + self.w(p + "attn.o_proj.weight") @ o

            h2 = _rms_norm(x, self.w(p + "mlp_norm.weight"), self.eps)
            g = self.w(p + "mlp.gate_proj.weight") @ h2
            u = self.w(p + "mlp.up_proj.weight") @ h2
            x = x + self.w(p + "mlp.down_proj.weight") @ (_silu(g) * u)

        x = _rms_norm(x, self.w("model.norm.weight"), self.eps)
        return self.lm_head(x)

    def lm_head(self, x):
        emb = self.w("model.embeddings.weight")
        if "lm_head.weight" in self._np:
            return self.w("lm_head.weight") @ x
        return emb @ x

    # -------------------------------------------------------------- 生成接口
    @staticmethod
    def sample(logits, temperature=1.0, top_k=0, seed=None):
        if temperature <= 0:
            return int(np.argmax(logits))
        z = logits.astype(np.float64) / temperature
        if top_k and top_k < z.size:
            thr = np.partition(z, -top_k)[-top_k]
            z = np.where(z < thr, -np.inf, z)
        z -= z.max()
        p = np.exp(z)
        p /= p.sum()
        rng = np.random.default_rng(seed)
        return int(rng.choice(z.size, p=p))

    def generate(self, ids, max_new=40, temperature=1.0, top_k=0, seed=None):
        state = self.reset(self.init_state())
        logits = None
        for t in ids:
            logits = self.step(int(t), state)
        out = list(int(i) for i in ids)
        for i in range(max_new):
            nxt = self.sample(logits, temperature, top_k, None if seed is None else seed + i)
            out.append(nxt)
            logits = self.step(nxt, state)
        return out