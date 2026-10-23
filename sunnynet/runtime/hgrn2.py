"""HGRN2 单步递推, 权重从 HF safetensors 读入.

数学依据: fla/layers/hgrn2.py 的 fused_recurrent 路径, 与 fla/ops/gla 的单步形式
共用同一个 recurrent 内核 (论文: HGRN2 与 GLA 递推结构相同, 仅参数化不同)。

递推 (逐 head 的 (K,V) 矩阵, 衰减只作用在 key 维):
    q = silu(q_proj(h))                         q_proj
    f = f_proj(h)
    g = logaddexp(lb, log1p(-lb) + logsigmoid(f))    lb = lower_bounds[i], i>0
    k = 1 - exp(g)                              写权重, 由衰减导出; 无 k_proj
    S_t = diag(exp(g)) S_{t-1} + k_t ⊗ i_t       i = i_proj(h) 即 value
    o_t = scale * S_t^T q_t
输出门: 无。o 在 value_dim 上做**整体** RMSNorm (非逐 head), 再 o_proj。

维度 (config.json): hidden 2048, expand_ratio 128 -> num_heads 16,
key_dim = value_dim = forget_dim = 2048, head_k_dim = head_v_dim = 128,
use_short_conv=false, use_lower_bound=true, tie_word_embeddings=false,
norm_eps 1e-6, hidden_ratio 4 -> intermediate 5632。

lb 由模型层循环预计算:
    lb_all = softmax(lower_bounds, dim=0).cumsum(0) - softmax(lower_bounds, dim=0)[0]
第 0 层 lb 恒为 0; softmax 在层维 (24) 上做, 不是通道维。

权重 2.7GB bf16 -> fp32 是 5.4GB, 本机空闲内存不足以整份驻留。
给 blob 路径时走 memmap (blob 是 fp32 且布局固定), 峰值只有逐层工作集;
否则退回逐张量惰性转换。
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np


def _silu(x):
    # x/(1+exp(-x)); x<<0 时 exp(-x) 溢出, 分岔到 x*exp(x)/(1+exp(x)) 等价且稳定
    pos = x >= 0
    out = np.empty_like(x)
    ex = np.exp(-np.abs(x), dtype=np.float32)
    out[pos] = x[pos] / (1.0 + ex[pos])
    out[~pos] = x[~pos] * ex[~pos] / (1.0 + ex[~pos])
    return out


def _logsigmoid(x):
    # log(sigmoid(x)) = -softplus(-x); 直接 exp(-x) 在 x<<0 时溢出
    return -np.logaddexp(0.0, -x).astype(np.float32)


def _rms_norm(x, w, eps):
    return x / np.sqrt((x * x).mean(axis=-1, keepdims=True, dtype=np.float32) + eps) * w


class _BlobReader:
    """从 C blob 读权重. 用 np.memmap, 不把 5.4GB fp32 读进内存。"""

    MAGIC = 0x32474853

    def __init__(self, path: str | Path):
        self.p = Path(path)
        with open(self.p, "rb") as f:
            head = f.read(36)
        (magic, ver, L, D, NH, HK, HV, _u) = struct.unpack("<8i", head[:32])
        if magic != self.MAGIC:
            raise ValueError(f"不是 HGRN2 blob (magic {magic:#x}): {self.p}")
        self.L, self.D, self.NH, self.HK, self.HV = L, D, NH, HK, HV
        self.KD = NH * HK
        self.VD = NH * HV
        with open(self.p, "rb") as f:
            f.seek(32)
            eps, scale, _x, lb_flag = struct.unpack("<4f", f.read(16))
            vocab, F, tied, _a, _b = struct.unpack("<5i", f.read(20))
        self.eps, self.scale = float(eps), float(scale)
        self.use_lb = lb_flag != 0.0
        self.vocab, self.F, self.tied = vocab, F, bool(tied)
        self._off = 32 + 16 + 20
        self.mm = np.memmap(self.p, dtype=np.float32, mode="r",
                            offset=self._off)
        self._build_index()

    def _take(self, n):
        """返回一段视图并前移游标 (float 个数)。"""
        v = self.mm[self._c:self._c + n]
        self._c += n
        return v

    def _build_index(self):
        L, D, KD, VD, F = self.L, self.D, self.KD, self.VD, self.F
        self._c = 0
        self.lb = []
        self.an = []
        self.qp = []
        self.fp = []
        self.ip = []
        self.op = []
        self.gn = []
        self.mn = []
        self.mgp = []
        self.mup = []
        self.mdp = []
        for _ in range(L):
            self.lb.append(self._take(D))
            self.an.append(self._take(D))
            self.qp.append(self._take(D * D))
            self.fp.append(self._take(D * D))
            self.ip.append(self._take(D * D))
            self.op.append(self._take(D * D))
            self.gn.append(self._take(D))
            self.mn.append(self._take(D))
            self.mgp.append(self._take(F * D))
            self.mup.append(self._take(F * D))
            self.mdp.append(self._take(D * F))
        self.embed = self._take(self.vocab * D)
        self.ln_f = self._take(D)
        self.head = None if self.tied else self._take(self.vocab * D)


class HGRN2:
    """HGRN2 单步推理; 每步 O(1) 与上下文长度无关."""

    def __init__(self, model_dir: str | Path, blob: str | Path | None = None):
        self.dir = Path(model_dir)
        b = Path(blob) if blob else self.dir / "hgrn2.c.bin"
        if b.is_file():
            self._blob = _BlobReader(b)
            self._src = None
            self._use_blob()
        else:
            self._blob = None
            self._load_hf()
            self._infer_dims()

    # ------------------------------------------------------------ blob 路径
    def _use_blob(self):
        b = self._blob
        self.D = b.D
        self.hidden = b.D
        self.vocab = b.vocab
        self.n_layers = b.L
        self.n_heads = b.NH
        self.head_f_dim = b.HK
        self.head_i_dim = b.HV
        self.forget_dim = b.KD
        self.input_dim = b.VD
        self.expand_ratio = b.HK
        self.eps = b.eps
        self.scale = b.scale
        self.use_lower_bound = b.use_lb
        self.tie = b.tied
        self.blob = b

    # -------------------------------------------------------------- HF 路径
    def _load_hf(self):
        import torch
        from safetensors import safe_open

        p = self.dir / "model.safetensors"
        if p.exists():
            self._h = safe_open(str(p), framework="pt")
            self._sd = None
        else:
            self._h = None
            self._sd = torch.load(self.dir / "pytorch_model.bin",
                                  map_location="cpu", weights_only=True)

    def w(self, name: str) -> np.ndarray:
        v = (self._h.get_tensor(name) if self._h is not None
             else self._sd[name])
        return v.to().float().numpy() if hasattr(v, "to") else v

    def _infer_dims(self):
        cp = self.dir / "config.json"
        cfg = json.loads(cp.read_text(encoding="utf-8")) if cp.exists() else {}
        emb = self.w("model.embeddings.weight")
        self.hidden = emb.shape[1]
        self.vocab = emb.shape[0]
        del emb
        self.n_layers = int(cfg.get("num_hidden_layers"))
        self.forget_dim = self.w("model.layers.0.attn.q_proj.weight").shape[0]
        self.input_dim = self.w("model.layers.0.attn.i_proj.weight").shape[0]
        self.expand_ratio = int(cfg.get("expand_ratio"))
        self.n_heads = self.forget_dim // self.expand_ratio
        self.head_f_dim = self.expand_ratio
        self.head_i_dim = self.input_dim // self.n_heads
        self.eps = float(cfg.get("norm_eps", cfg.get("rms_norm_eps", 1e-6)))
        self.scale = self.head_f_dim ** -0.5
        self.use_lower_bound = bool(cfg.get("use_lower_bound", False))
        self.tie = bool(cfg.get("tie_word_embeddings", False))

    def lower_bounds(self) -> np.ndarray:
        """层循环里预算好的 lb[L, D]: softmax 在层维上做再 cumsum, 首层为 0。"""
        if self._blob is not None:
            return np.stack([np.asarray(x) for x in self._blob.lb])
        lb = self.w("model.lower_bounds")
        sm = np.exp(lb - lb.max(axis=0, keepdims=True))
        sm = sm / sm.sum(axis=0, keepdims=True)
        return (np.cumsum(sm, axis=0) - sm[0]).astype(np.float32)

    # ------------------------------------------------------------------ 状态
    def init_state(self):
        L, H, K, V = self.n_layers, self.n_heads, self.head_f_dim, self.head_i_dim
        return {"S": np.zeros((L, H, K, V), dtype=np.float32)}

    def reset(self, state):
        for v in state.values():
            v[...] = 0.0
        return state

    # ------------------------------------------------------------------ 前向
    def _layer_w(self, i):
        """逐层权重: blob 走 memmap 视图 (按 (out,in) reshape), HF 走按名取。"""
        if self._blob is not None:
            b = self._blob
            D, KD, VD, F = self.D, b.KD, b.VD, b.F
            return {"lb": b.lb[i], "an": b.an[i],
                    "qp": b.qp[i].reshape(KD, D), "fp": b.fp[i].reshape(KD, D),
                    "ip": b.ip[i].reshape(VD, D), "op": b.op[i].reshape(D, VD),
                    "gn": b.gn[i], "mn": b.mn[i],
                    "mgp": b.mgp[i].reshape(F, D), "mup": b.mup[i].reshape(F, D),
                    "mdp": b.mdp[i].reshape(D, F)}
        p = f"model.layers.{i}."
        return {"lb": None, "an": self.w(p + "attn_norm.weight"),
                "qp": self.w(p + "attn.q_proj.weight"),
                "fp": self.w(p + "attn.f_proj.weight"),
                "ip": self.w(p + "attn.i_proj.weight"),
                "op": self.w(p + "attn.o_proj.weight"),
                "gn": self.w(p + "attn.g_norm.weight"),
                "mn": self.w(p + "mlp_norm.weight"),
                "mgp": self.w(p + "mlp.gate_proj.weight"),
                "mup": self.w(p + "mlp.up_proj.weight"),
                "mdp": self.w(p + "mlp.down_proj.weight")}

    def step(self, token: int, state) -> np.ndarray:
        H, K, V = self.n_heads, self.head_f_dim, self.head_i_dim
        S_all = state["S"]
        lb_all = self.lower_bounds() if self.use_lower_bound else None

        emb = (self._blob.embed if self._blob is not None
               else self.w("model.embeddings.weight").reshape(-1))
        x = np.array(emb[token * self.hidden:(token + 1) * self.hidden],
                     dtype=np.float32)

        for i in range(self.n_layers):
            W = self._layer_w(i)
            res = x
            h = _rms_norm(x, W["an"], self.eps)

            q = _silu(W["qp"] @ h)
            f = W["fp"] @ h
            v = W["ip"] @ h

            g = _logsigmoid(f)
            if lb_all is not None and i > 0:
                lb = lb_all[i]
                g = np.logaddexp(np.log(lb), np.log1p(-lb) + g).astype(np.float32)
            k = (1.0 - np.exp(g, dtype=np.float32)).astype(np.float32)

            q2 = q.reshape(H, K)
            k2 = k.reshape(H, K)
            v2 = v.reshape(H, V)
            decay = np.exp(g.reshape(H, K), dtype=np.float32)

            S = S_all[i] * decay[:, :, None] + k2[:, :, None] * v2[:, None, :]
            S_all[i] = S

            o = np.einsum("hkv,hk->hv", S, q2) * self.scale
            o = _rms_norm(o.reshape(-1), W["gn"], self.eps)
            x = res + W["op"] @ o

            h2 = _rms_norm(x, W["mn"], self.eps)
            x = x + W["mdp"] @ (
                _silu(W["mgp"] @ h2) * (W["mup"] @ h2))

        ln_f = (self._blob.ln_f if self._blob is not None
                else self.w("model.norm.weight"))
        x = _rms_norm(x, ln_f, self.eps)
        return self.lm_head(x)

    def lm_head(self, x):
        if self._blob is not None:
            W = self._blob.embed if self._blob.tied else self._blob.head
            return W.reshape(self.vocab, self.hidden) @ x
        if not self.tie and "lm_head.weight" in (
                set(self._h.keys()) if self._h is not None else set(self._sd)):
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
            nxt = self.sample(logits, temperature, top_k,
                              None if seed is None else seed + i)
            out.append(nxt)
            logits = self.step(nxt, state)
        return out