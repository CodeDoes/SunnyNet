"""GLA (Gated Linear Attention) 单步递推, 权重从 HF safetensors 读入.

数学依据: fla/layers/gla.py 的 fused_recurrent 路径
与 fla/ops/common/fused_recurrent.py 的单步形式。

状态是逐 head 的 (k_dim, v_dim) 矩阵, 递推为:
    S_t = diag(exp(gk_t)) S_{t-1} + k_t ⊗ v_t
    o_t = scale * S_t^T q_t
其中 gk = logsigmoid(gk_proj(h)) / gate_logit_normalizer 是**只作用在 key 维**的
对数衰减, 衰减因此可分解为按行缩放 (无 value 维耦合), 不需要矩阵值递推层级。
本 checkpoint use_gv=false, 故 value 维无衰减。

维度 (config.json): hidden 1024, num_heads 4, expand_k 0.5, expand_v 1.0
  -> key_dim 512, value_dim 1024, head_k_dim 128, head_v_dim 256
  use_output_gate=true, gate_fn=swish, fuse_norm=true, norm_eps 1e-6,
  gate_logit_normalizer 16, use_short_conv=false, tie_word_embeddings=true。
输出门: y = RMSNorm(o) * weight * silu(g_proj(h)), 逐 head 在 head_v_dim 上归一化。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _silu(x):
    return x / (1.0 + np.exp(-x, dtype=np.float32))


def _logsigmoid(x):
    # log(sigmoid(x)) = -softplus(-x); 直接 exp(-x) 在 x<<0 时溢出
    return -np.logaddexp(0.0, -x).astype(np.float32)


def _rms_norm(x, w, eps):
    return x / np.sqrt((x * x).mean(axis=-1, keepdims=True, dtype=np.float32) + eps) * w


class GLA:
    """GLA 单步推理; 每步 O(1) 与上下文长度无关."""

    def __init__(self, model_dir: str | Path):
        self.dir = Path(model_dir)
        self._load()
        self._infer_dims()

    # ------------------------------------------------------------------ 加载
    def _load(self):
        import torch
        from safetensors.torch import load_file

        p = self.dir / "model.safetensors"
        if p.exists():
            sd = load_file(str(p))
        else:
            sd = torch.load(self.dir / "pytorch_model.bin",
                            map_location="cpu", weights_only=True)
        self._np = {k: v.detach().to(torch.float32).numpy() for k, v in sd.items()}

    def w(self, name: str) -> np.ndarray:
        return self._np[name]

    def _infer_dims(self):
        cfg = {}
        cp = self.dir / "config.json"
        if cp.exists():
            import json
            cfg = json.loads(cp.read_text(encoding="utf-8"))
        self.hidden = self.w("model.embeddings.weight").shape[1]
        self.vocab = self.w("model.embeddings.weight").shape[0]
        self.n_layers = len({k.split(".")[2] for k in self._np
                             if k.startswith("model.layers.")})
        self.key_dim = self.w("model.layers.0.attn.q_proj.weight").shape[0]
        self.value_dim = self.w("model.layers.0.attn.v_proj.weight").shape[0]
        if "num_heads" in cfg:
            self.n_heads = int(cfg["num_heads"])
        else:
            # expand_k 可确定 head_k_dim 与 num_heads 的乘积; 无 config 时无解,
            # 退回用 gk 低秩投影的行数不可靠, 直接报错要求显式 config。
            raise ValueError("config.json 缺 num_heads, 无法定头数")
        self.head_k_dim = self.key_dim // self.n_heads
        self.head_v_dim = self.value_dim // self.n_heads
        self.gate_rank = self.w("model.layers.0.attn.gk_proj.0.weight").shape[0]
        self.gate_logit_normalizer = float(cfg.get("gate_logit_normalizer", 16))
        self.clamp_min = cfg.get("clamp_min", None)
        self.eps = float(cfg.get("norm_eps", 1e-6))
        self.scale = self.head_k_dim ** -0.5
        self.use_output_gate = bool(cfg.get("use_output_gate", True))
        self.tie = bool(cfg.get("tie_word_embeddings", True))
        exp_k = cfg.get("expand_k")
        if exp_k is not None:
            assert self.key_dim == round(self.hidden * float(exp_k)), \
                (self.key_dim, self.hidden, exp_k)

    # ------------------------------------------------------------------ 状态
    def init_state(self):
        L, H, K, V = self.n_layers, self.n_heads, self.head_k_dim, self.head_v_dim
        return {"S": np.zeros((L, H, K, V), dtype=np.float32)}

    def reset(self, state):
        for v in state.values():
            v[...] = 0.0
        return state

    # ------------------------------------------------------------------ 前向
    def step(self, token: int, state) -> np.ndarray:
        H, K, V = self.n_heads, self.head_k_dim, self.head_v_dim
        S_all = state["S"]

        x = self.w("model.embeddings.weight")[token].astype(np.float32)

        for i in range(self.n_layers):
            p = f"model.layers.{i}."
            res = x
            h = _rms_norm(x, self.w(p + "attn_norm.weight"), self.eps)

            q = self.w(p + "attn.q_proj.weight") @ h
            k = self.w(p + "attn.k_proj.weight") @ h
            v = self.w(p + "attn.v_proj.weight") @ h
            gk = (self.w(p + "attn.gk_proj.1.weight")
                  @ (self.w(p + "attn.gk_proj.0.weight") @ h)
                  + self.w(p + "attn.gk_proj.1.bias"))
            gk = _logsigmoid(gk) / self.gate_logit_normalizer
            if self.clamp_min is not None:
                gk = np.maximum(gk, self.clamp_min)

            q2 = q.reshape(H, K)
            k2 = k.reshape(H, K)
            v2 = v.reshape(H, V)
            decay = np.exp(gk.reshape(H, K), dtype=np.float32)

            S = S_all[i] * decay[:, :, None] + k2[:, :, None] * v2[:, None, :]
            S_all[i] = S

            o = np.einsum("hkv,hk->hv", S, q2) * self.scale
            g = self.w(p + "attn.g_proj.weight") @ h
            o = _rms_norm(o, self.w(p + "attn.g_norm_swish_gate.weight"), self.eps)
            o = (o * _silu(g.reshape(H, V))).reshape(-1)
            x = res + self.w(p + "attn.o_proj.weight") @ o

            h2 = _rms_norm(x, self.w(p + "mlp_norm.weight"), self.eps)
            x = x + self.w(p + "mlp.down_proj.weight") @ (
                _silu(self.w(p + "mlp.gate_proj.weight") @ h2)
                * (self.w(p + "mlp.up_proj.weight") @ h2))

        x = _rms_norm(x, self.w("model.norm.weight"), self.eps)
        return self.lm_head(x)

    def lm_head(self, x):
        if not self.tie and "lm_head.weight" in self._np:
            return self.w("lm_head.weight") @ x
        return self.w("model.embeddings.weight") @ x

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