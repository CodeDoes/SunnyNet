"""GGUF -> GGUF 量化工具, 并给出困惑度评测.

量化在 Python 侧完成: 用 gguf.quants 把 fp32 权重压成目标类型, 再以
`raw_dtype` 声明把原始字节写回 GGUF (llama.cpp 的 gguf-writer 也是这个路子).

支持写出的类型: Q4_0 / Q4_1 / Q5_0 / Q5_1 / Q8_0 (gguf-py 的 quantize 只实现了这些).
**K-quant (Q4_K_M / Q5_K_M / Q6_K) 只能读不能写** —— gguf-py 的 quantize 对它们
抛 NotImplementedError, 本工具会明确报错而不是静默降级.

只量化 2 维权重矩阵; 1 维向量与词表嵌入保持原精度 (嵌入是查表, 量化收益低误差大).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from .gguf import GGUFModel

# 名称 -> GGML 类型名; 别名按 llama.cpp 的习惯
_TYPES = {
    "q4_0": "Q4_0", "q4_1": "Q4_1", "q5_0": "Q5_0", "q5_1": "Q5_1",
    "q8_0": "Q8_0",
    # 读得进但写不出的类型, 单独列出让报错信息有用
}
_READONLY = {"q4_k": "Q4_K", "q4_k_m": "Q4_K", "q5_k": "Q5_K",
             "q5_k_m": "Q5_K", "q6_k": "Q6_K", "q2_k": "Q2_K",
             "q3_k": "Q3_K", "q8_k": "Q8_K", "iq4_nl": "IQ4_NL"}

# 这些名字的张量不量化 (嵌入查表 / 归一化 / 偏置)
_SKIP_SUFFIX = ("_norm.weight", "_norm.bias", ".bias", "embed.weight",
                "token_embd.weight", ".scale", ".alpha")


def _skip(name: str, arr: np.ndarray) -> bool:
    if arr.ndim != 2:
        return True
    if name.endswith(_SKIP_SUFFIX):
        return True
    return False


def resolve_type(bits: str | int) -> str:
    """把 --bits 4 / 8 或 --type q8_0 统一成量化类型名."""
    s = str(bits).lower().strip()
    if s in _TYPES:
        return _TYPES[s]
    if s in _READONLY:
        raise SystemExit(
            f"类型 {s} 在 gguf-py 里只能读不能写 (quantize 未实现 K-quant)。\n"
            f"可写类型: {', '.join(sorted(_TYPES))}")
    if s in ("4", "int4"):
        return "Q4_0"
    if s in ("8", "int8"):
        return "Q8_0"
    if s in ("5", "int5"):
        return "Q5_0"
    raise SystemExit(f"未知量化类型 {bits!r}; 可写: {', '.join(sorted(_TYPES))}")


def quantize_gguf(src: str | Path, dst: str | Path, qtype: str,
                  skip_embed: bool = True) -> dict:
    """把 GGUF 的 2 维权重矩阵量化到 qtype, 其余按原精度写回."""
    from gguf import GGUFWriter, quants
    from gguf.constants import GGMLQuantizationType as QT

    t = QT[qtype]
    g = GGUFModel(src)
    w = GGUFWriter(str(dst), arch="")

    # 元数据原样搬过去 (含架构超参、词表)
    for k, v in g.meta.items():
        if k.startswith("GGUF.") or v is None:
            continue
        try:
            if isinstance(v, bool):
                w.add_bool(k, v)
            elif isinstance(v, int):
                w.add_uint32(k, v)
            elif isinstance(v, float):
                w.add_float32(k, v)
            elif isinstance(v, str):
                w.add_string(k, v)
            elif isinstance(v, list) and v and isinstance(v[0], str):
                w.add_array(k, [str(x) for x in v])
            elif isinstance(v, list) and v and isinstance(v[0], int):
                w.add_array(k, [int(x) for x in v])
            elif isinstance(v, list) and v and isinstance(v[0], float):
                w.add_array(k, [float(x) for x in v])
        except Exception:
            pass

    stats = {"quantized": 0, "kept": 0, "src_bytes": 0, "dst_bytes": 0}
    for name in g.names():
        arr = g.tensor(name)
        if arr.size == 0:
            continue
        if _skip(name, arr):
            w.add_tensor(name, arr.astype(np.float32))
            stats["kept"] += 1
            stats["dst_bytes"] += arr.astype(np.float32).nbytes
        else:
            q = quants.quantize(np.ascontiguousarray(arr, dtype=np.float32), t)
            # 传字节形状: gguf-py 由 (rows, 每行字节数) 反推逻辑形状
            w.add_tensor(name, q, raw_dtype=t)
            stats["quantized"] += 1
            stats["dst_bytes"] += q.nbytes
        stats["src_bytes"] += arr.nbytes

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    out = Path(dst)
    return {"file": str(out), "bytes": out.stat().st_size,
            "mb": round(out.stat().st_size / 2**20, 1), "type": qtype, **stats}


def perplexity(model: str | Path, text_file: str | Path | None = None,
               max_tokens: int = 512, gguf_backend: str = "numpy",
               quant: str = "int4") -> dict:
    """在给定文本上算逐 token 困惑度 (仅 RWKV-7 GGUF).

    无 text_file 时用内置的一段英文/中文混合文本, 保证任何机器都能跑出数.
    """
    if gguf_backend == "c":
        from .rwkv7_c import SnRWKV7
        from .blob import ensure_rwkv7_blob
        from .rwkv7 import RWKV7
        rk = RWKV7(model)
        bp = ensure_rwkv7_blob(model, quant=quant)
        h = SnRWKV7(bp, rk.vocab)

        def step(tok, _state):
            return h.step_np(int(tok))
        tok = rk.tok
        vocab = rk.vocab
    else:
        from .rwkv7 import RWKV7
        rk = RWKV7(model)
        tok = rk.tok
        vocab = rk.vocab
        state = rk.init_state()
        _s = {"st": state}

        def step(t, _state):
            return rk.step(int(t), _s["st"])
    ids = tok.encode(_read_text(text_file))[:max_tokens]
    if len(ids) < 8:
        return {"tokens": len(ids), "ppl": float("nan")}
    logits = None
    nll = 0.0
    for i, t in enumerate(ids):
        if logits is not None:
            z = logits.astype(np.float64)
            z -= z.max()
            lse = np.log(np.exp(z).sum())
            nll += float(lse - z[int(t)])
        logits = np.asarray(step(t, None), dtype=np.float32)
    n = len(ids) - 1
    return {"tokens": len(ids), "ppl": float(np.exp(nll / max(n, 1)))}


_DEFAULT_TEXT = (
    "The recurrence keeps a state of fixed size, so the cost of one step does "
    "not grow with the context length. 递推状态大小固定, 单步开销不随上下文增长。"
    "A linear recurrence layer replaces attention with a running summary, and "
    "the summary is updated by a small number of element-wise operations."
)


def _read_text(p) -> str:
    if p is None:
        return _DEFAULT_TEXT
    return Path(p).read_text(encoding="utf-8", errors="replace")


def quantize_main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="sunnynet quantize",
                                 description="GGUF 量化 (Q4_0/Q5_0/Q8_0 等)")
    ap.add_argument("src", help="输入 GGUF")
    ap.add_argument("dst", help="输出 GGUF")
    ap.add_argument("--bits", default=None, help="4 / 5 / 8")
    ap.add_argument("--type", default=None, help="量化类型名 (q4_0/q5_0/q8_0 ...)")
    ap.add_argument("--ppl", action="store_true", help="量化前后各算一次困惑度")
    ap.add_argument("--ppl-text", default=None, help="困惑度评测文本文件")
    ap.add_argument("--ppl-tokens", type=int, default=256)
    ap.add_argument("--gguf-backend", choices=["numpy", "c"], default="numpy",
                    help="困惑度评测用的后端")
    args = ap.parse_args(argv)

    spec = args.type or args.bits or "8"
    qtype = resolve_type(spec)
    t0 = time.perf_counter()
    info = quantize_gguf(args.src, args.dst, qtype)
    info["seconds"] = round(time.perf_counter() - t0, 1)
    src_mb = Path(args.src).stat().st_size / 2**20
    info["ratio"] = round(src_mb / max(info["mb"], 1e-9), 3)
    if args.ppl:
        p0 = perplexity(args.src, args.ppl_text, args.ppl_tokens,
                        args.gguf_backend)
        p1 = perplexity(args.dst, args.ppl_text, args.ppl_tokens,
                        args.gguf_backend)
        info["ppl_src"] = p0["ppl"]
        info["ppl_dst"] = p1["ppl"]
        if p0["ppl"] > 0:
            info["ppl_delta_pct"] = round(
                (p1["ppl"] - p0["ppl"]) / p0["ppl"] * 100, 2)
    print(json.dumps(info, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(quantize_main())