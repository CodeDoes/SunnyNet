"""RWKV-7 单步递推, 权重从 GGUF 读入, 不依赖 torch.

数学依据: BlinkDL/RWKV-LM 的 RWKV-v7/rwkv_v7_numpy.py (johanwind 的纯 numpy 参考)
与 llama.cpp 的 ggml_rwkv_wkv7 内核 (src/models/rwkv7-base.cpp).

状态是逐 head 的矩阵 S[h] (HEAD_SIZE, HEAD_SIZE), 更新含 rank-1 校正:
    S_t = S_{t-1} * w_t - outer(S_{t-1} kk_t, kk_t * a_t) + outer(v_t, k_t)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .gguf import GGUFModel


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x, dtype=np.float32))


def _layer_norm(x, w, b, eps):
    return (x - x.mean(dtype=np.float32)) / np.sqrt(x.var(dtype=np.float32) + eps) * w + b


def _group_norm(x, w, b, eps):
    """逐 head 归一化 (沿 head_size 维), 再压平到 (n_embd,)."""
    m = x.mean(axis=-1, keepdims=True, dtype=np.float32)
    v = x.var(axis=-1, keepdims=True, dtype=np.float32)
    n = ((x - m) / np.sqrt(v + eps)).reshape(-1)
    return n * w + b


class _Tok:
    """从 GGUF 词表构造的贪心最长匹配 tokenizer (RWKV 用字节级 trie)."""

    def __init__(self, tokens: list[bytes]):
        self.idx2tok = {i: t for i, t in enumerate(tokens)}
        self.root: dict = {}
        for i, t in self.idx2tok.items():
            if not t:
                continue
            node = self.root
            for ch in t:
                node = node.setdefault(ch, {})
            node[-1] = i

    def encode(self, s: str) -> list[int]:
        data = s.encode("utf-8")
        out, i = [], 0
        n = len(data)
        while i < n:
            node, j, best, best_id = self.root, i, None, None
            while j < n and data[j] in node:
                node = node[data[j]]
                j += 1
                if -1 in node:
                    best, best_id = j, node[-1]
            if best_id is None:
                best, best_id = i + 1, 0
            out.append(best_id)
            i = best
        return out

    def decode(self, ids: list[int]) -> str:
        raw = b"".join(self.idx2tok.get(i, b"") for i in ids)
        return raw.decode("utf-8", errors="replace")


class RWKV7:
    """单步 RNN 推理; 每步 O(1) 与上下文长度无关."""

    def __init__(self, gguf_path: str | Path, threads: int = 0):
        self.g = GGUFModel(gguf_path)
        self.n_layer = self.g.get_int("rwkv7.block_count", 12)
        self.n_embd = self.g.get_int("rwkv7.embedding_length", 768)
        self.head_size = self.g.get_int("rwkv7.wkv.head_size", 64)
        self.n_head = self.n_embd // self.head_size
        self.vocab = len(self.g.tokens())
        self.eps = self.g.get_float("rwkv7.attention.layer_norm_epsilon", 1e-5)
        self.eps_group = 64e-5
        if threads:
            np.set_printoptions()
        self.tok = _Tok(self.g.tokens())
        self._preload()

    def _preload(self):
        """一次性反量化到 fp32 并缓存, 避免每步重复解码."""
        names = list(self.g.names())
        for n in names:
            self.g.tensor(n)

    # ---------------------------------------------------------------- 权重取用
    # GGUF 把 LoRA 分解矩阵存成 (out, in), 与官方 .pth 相反; 其余矩阵方向一致.
    # 数值锚定见 experiments/_gguf_align.py (逐张量 max|diff| = 0).
    _TRANSPOSE = ("time_mix_w1.weight", "time_mix_w2.weight",
                  "time_mix_a1.weight", "time_mix_a2.weight",
                  "time_mix_g1.weight", "time_mix_g2.weight",
                  "time_mix_v1.weight", "time_mix_v2.weight")

    def _T(self, name: str) -> np.ndarray:
        a = self.g.tensor(name)
        if name.endswith(self._TRANSPOSE):
            a = a.T
        return np.ascontiguousarray(a, dtype=np.float32)

    def _vec(self, name: str) -> np.ndarray:
        return self.g.tensor(name).astype(np.float32, copy=False).reshape(-1)

    # ---------------------------------------------------------------- 前向
    def init_state(self):
        S = np.zeros((self.n_layer, self.n_head, self.head_size, self.head_size),
                     dtype=np.float32)
        shift = np.zeros((self.n_layer, 2, self.n_embd), dtype=np.float32)
        return {"S": S, "shift": shift}

    def reset(self, state):
        state["S"][...] = 0.0
        state["shift"][...] = 0.0
        return state

    def step(self, token: int, state) -> np.ndarray:
        EPS = self.eps
        HS, NH, D = self.head_size, self.n_head, self.n_embd
        S_all, shift = state["S"], state["shift"]

        x = self._T("token_embd.weight")[token].copy()
        x = _layer_norm(x, self._vec("token_embd_norm.weight"),
                        self._vec("token_embd_norm.bias"), EPS)

        v_first = None
        for i in range(self.n_layer):
            p = f"blk.{i}."
            # ---- time mixing
            xn = _layer_norm(x, self._vec(p + "attn_norm.weight"),
                             self._vec(p + "attn_norm.bias"), EPS)
            last = shift[i, 0]
            lerp = self.g.tensor(p + "time_mix_lerp_fused.weight").reshape(6, D)
            d = last - xn
            xr = xn + lerp[0] * d
            xw = xn + lerp[1] * d
            xk = xn + lerp[2] * d
            xv = xn + lerp[3] * d
            xa = xn + lerp[4] * d
            xg = xn + lerp[5] * d

            r = self._T(p + "time_mix_receptance.weight") @ xr
            w = np.exp(-_sigmoid(np.tanh(xw @ self._T(p + "time_mix_w1.weight"))
                                 @ self._T(p + "time_mix_w2.weight")
                                 + self._vec(p + "time_mix_w0")) / np.sqrt(np.e))
            k = self._T(p + "time_mix_key.weight") @ xk
            v = self._T(p + "time_mix_value.weight") @ xv
            if v_first is None:
                v_first = v
            else:
                mix = _sigmoid((xv @ self._T(p + "time_mix_v1.weight"))
                               @ self._T(p + "time_mix_v2.weight")
                               + self._vec(p + "time_mix_v0"))
                v = v + (v_first - v) * mix
            a = _sigmoid((xa @ self._T(p + "time_mix_a1.weight"))
                         @ self._T(p + "time_mix_a2.weight")
                         + self._vec(p + "time_mix_a0"))
            g = _sigmoid(xg @ self._T(p + "time_mix_g1.weight")) @ \
                self._T(p + "time_mix_g2.weight")

            kk = k * self._vec(p + "time_mix_k_k")
            k = k + k * (a - 1.0) * self._vec(p + "time_mix_k_a")

            r2 = r.reshape(NH, HS)
            w2 = w.reshape(NH, HS)
            k2 = k.reshape(NH, HS)
            v2 = v.reshape(NH, HS)
            a2 = a.reshape(NH, HS)
            rk2 = self._vec(p + "time_mix_r_k").reshape(NH, HS)
            kk2 = kk.reshape(NH, HS)
            kk2 = kk2 / np.maximum(np.linalg.norm(kk2, axis=1, keepdims=True), 1e-12)

            S = S_all[i]
            # rank-1 去除项用未衰减的 S (与参考实现一致)
            proj = np.einsum("hij,hj->hi", S, kk2)
            S = S * w2[:, None, :] \
                - proj[:, :, None] * (kk2 * a2)[:, None, :] \
                + v2[:, :, None] * k2[:, None, :]
            S_all[i] = S

            y = np.einsum("hij,hj->hi", S, r2)
            y = _group_norm(y, self._vec(p + "time_mix_ln.weight"),
                            self._vec(p + "time_mix_ln.bias"), self.eps_group)
            y = y + (((r2 * k2 * rk2).sum(axis=1, keepdims=True) * v2).reshape(-1))
            y = y * g
            dx = self._T(p + "time_mix_output.weight") @ y
            x = x + dx
            shift[i, 0] = xn

            # ---- channel mixing
            xn2 = _layer_norm(x, self._vec(p + "attn_norm_2.weight"),
                              self._vec(p + "attn_norm_2.bias"), EPS)
            last2 = shift[i, 1]
            xk2 = xn2 + self._vec(p + "channel_mix_lerp_k") * (last2 - xn2)
            ck = np.maximum(self._T(p + "channel_mix_key.weight") @ xk2, 0.0)
            dc = self._T(p + "channel_mix_value.weight") @ (ck * ck)
            x = x + dc
            shift[i, 1] = xn2

        x = _layer_norm(x, self._vec("output_norm.weight"),
                        self._vec("output_norm.bias"), EPS)
        logits = self._T("output.weight") @ x
        if not self.g.has("output.weight"):
            logits = self._T("token_embd.weight") @ x
        return logits

    @staticmethod
    def sample(logits: np.ndarray, temperature: float = 1.0,
               top_k: int = 0, seed: int | None = None) -> int:
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

    def generate(self, prompt: str, max_new: int = 40, temperature: float = 1.0,
                 top_k: int = 0, seed: int | None = None):
        ids = self.tok.encode(prompt)
        state = self.reset(self.init_state())
        logits = None
        for t in ids:
            logits = self.step(t, state)
        out = list(ids)
        for i in range(max_new):
            nxt = self.sample(logits, temperature, top_k,
                              None if seed is None else seed + i)
            out.append(nxt)
            logits = self.step(nxt, state)
        return self.tok.decode(out), out