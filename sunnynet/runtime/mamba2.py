"""Mamba-2 单步递推, 权重从 HF checkpoint 读入, 不依赖 torch.

数学依据: transformers 的 models/mamba2/modeling_mamba2.py
(见 mamba2_selective_state_update 与 MambaRMSNormGated)。

每 head 状态是 (head_dim, state_size) 矩阵, 但衰减是外积可分解的:
    dA = exp(dt_h * A_h)                     # 标量, 对 (p,n) 全同
    S_t = S_{t-1} * dA + (dt_h * B_g) ⊗ x_h  # 仍是一阶递推, 无矩阵-向量乘积
因此不需要矩阵值递推层级, 可复用元素级 SN_FMA。

维度来自 shape 推断 (config.json 不含 ssm 参数):
    d_model=768, expand=2 -> d_inner=1536, num_heads=24, head_dim=64,
    n_groups=8, state_size=16, conv_kernel=4, conv_dim=1792.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _silu(x):
    return x / (1.0 + np.exp(-x, dtype=np.float32))


def _softplus(x):
    # 数值稳定: log1p(exp(-|x|)) + max(x, 0)
    return np.log1p(np.exp(-np.abs(x), dtype=np.float32)) + np.maximum(x, 0.0)


def _rms_norm(x, w, eps):
    return x / np.sqrt((x * x).mean(dtype=np.float32) + eps) * w


class Mamba2:
    """单步 RNN 推理; 每步 O(1) 与上下文长度无关."""

    def __init__(self, model_dir: str | Path, n_groups: int | None = None,
                 state_size: int | None = None):
        self.dir = Path(model_dir)
        self._ng_override = n_groups
        self._ns_override = state_size
        self._load()
        self._infer_dims()

    # ------------------------------------------------------------------ 加载
    def _load(self):
        import torch

        p = self.dir / "pytorch_model.bin"
        if p.exists():
            self.sd = torch.load(p, map_location="cpu", weights_only=True)
        else:
            from safetensors.torch import load_file
            self.sd = load_file(str(self.dir / "model.safetensors"))
        self._np = {}
        for k, v in self.sd.items():
            self._np[k] = v.detach().to(torch.float32).numpy()

    def w(self, name: str) -> np.ndarray:
        """张量取用; 权重在 checkpoint 中为 (out, in), 转置成 (in, out) 供 x @ W 使用."""
        if name not in self._np:
            raise KeyError(name)
        return self._np[name]

    def _infer_dims(self):
        self.d_model = self.w("backbone.embedding.weight").shape[1]
        self.n_heads = self.w("backbone.layers.0.mixer.A_log").shape[0]
        self.n_layers = len({k.split(".")[2] for k in self.sd if k.startswith("backbone.layers.")})
        # out_proj.weight 是 (d_model, d_inner)
        self.d_inner = self.w("backbone.layers.0.mixer.out_proj.weight").shape[1]
        self.head_dim = self.d_inner // self.n_heads
        self.conv_dim = self.w("backbone.layers.0.mixer.conv1d.weight").shape[0]
        self.conv_kernel = self.w("backbone.layers.0.mixer.conv1d.weight").shape[-1]
        # conv_dim = d_inner + 2*n_groups*state_size, 只能定出乘积;
        # 拆分取 mamba_ssm 的 Mamba2 默认 ngroups=1, d_state=128 (与 A_log/nheads 自洽)
        prod = (self.conv_dim - self.d_inner) // 2
        if self._ns_override is not None:
            self.state_size = self._ns_override
            self.n_groups = prod // self.state_size
        elif self._ng_override is not None:
            self.n_groups = self._ng_override
            self.state_size = prod // self.n_groups
        else:
            self.n_groups = 1
            self.state_size = prod
        assert self.n_groups * self.state_size == prod, (self.n_groups, self.state_size, prod)
        self.eps = 1e-5
        self.dt_limit = (0.0, float("inf"))
        self.vocab = self.w("backbone.embedding.weight").shape[0]

    # ------------------------------------------------------------------ 状态
    def init_state(self):
        L = self.n_layers
        return {
            # 每 head 的 (head_dim, state_size) 状态
            "S": np.zeros((L, self.n_heads, self.head_dim, self.state_size), dtype=np.float32),
            # 深度卷积的 (conv_kernel-1) 步历史, 存 (conv_dim,)
            "conv": np.zeros((L, self.conv_kernel - 1, self.conv_dim), dtype=np.float32),
        }

    def reset(self, state):
        state["S"][...] = 0.0
        state["conv"][...] = 0.0
        return state

    # ------------------------------------------------------------------ 前向
    def step(self, token: int, state) -> np.ndarray:
        NH, HD, NS, NG = self.n_heads, self.head_dim, self.state_size, self.n_groups
        L = self.n_layers
        S_all, conv_all = state["S"], state["conv"]

        x = self.w("backbone.embedding.weight")[token].astype(np.float32)

        for i in range(L):
            p = f"backbone.layers.{i}."
            res = x
            h = _rms_norm(x, self.w(p + "norm.weight"), self.eps)

            proj = self.w(p + "mixer.in_proj.weight") @ h          # (3352,)
            gate = proj[: self.d_inner]
            bc = proj[self.d_inner: self.d_inner + self.conv_dim]
            dt = proj[self.d_inner + self.conv_dim:]

            # 深度因果卷积, kernel=k, 含历史
            hist = conv_all[i]                                     # (k-1, conv_dim)
            kw = self.w(p + "mixer.conv1d.weight").reshape(self.conv_dim, self.conv_kernel)
            kb = self.w(p + "mixer.conv1d.bias")
            acc = np.zeros(self.conv_dim, dtype=np.float32)
            for t in range(self.conv_kernel - 1):
                acc += kw[:, t] * hist[t]
            acc += kw[:, -1] * bc
            # 历史滑动: 丢最旧, 追加当前
            if self.conv_kernel > 1:
                hist[:-1] = hist[1:]
                hist[-1] = bc
            act = _silu(acc + kb)

            x_ssm = act[: self.d_inner].reshape(NH, HD)
            B = act[self.d_inner: self.d_inner + NG * NS].reshape(NG, NS)
            C = act[self.d_inner + NG * NS:].reshape(NG, NS)

            A = -np.exp(self.w(p + "mixer.A_log"))
            dth = _softplus(dt + self.w(p + "mixer.dt_bias"))
            dth = np.clip(dth, self.dt_limit[0], self.dt_limit[1])

            dA = np.exp(dth[:, None] * A[:, None])                 # (NH, HD) 逐元素
            Bg = np.repeat(B, NH // NG, axis=0)                    # (NH, NS)
            Cg = np.repeat(C, NH // NG, axis=0)
            dBx = (dth[:, None] * Bg)[:, None, :] * x_ssm[:, :, None]

            S = S_all[i] * dA[:, :, None] + dBx
            S_all[i] = S

            y = np.einsum("hpn,hn->hp", S, Cg)
            y = y + self.w(p + "mixer.D")[:, None] * x_ssm
            y = y.reshape(-1)
            y = y * _silu(gate)
            y = _rms_norm(y, self.w(p + "mixer.norm.weight"), self.eps)
            x = res + self.w(p + "mixer.out_proj.weight") @ y

        x = _rms_norm(x, self.w("backbone.norm_f.weight"), self.eps)
        return self.lm_head(x)

    def lm_head(self, x):
        emb = self.w("backbone.embedding.weight")
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