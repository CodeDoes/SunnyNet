"""GGUF 只读加载 + RWKV-7 单步递推.

张量命名按 llama.cpp 的 rwkv7 实现 (src/models/rwkv7*.cpp). 参考数学见
BlinkDL/RWKV-LM 的 RWKV-v7/rwkv_v7_numpy.py 与 ggml_rwkv_wkv7 内核.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

try:
    from gguf import GGUFReader
    from gguf.constants import GGMLQuantizationType as QT
except ImportError:  # pragma: no cover
    GGUFReader = None
    QT = None


_MAGIC = b"GGUF"


def available() -> bool:
    return GGUFReader is not None


def _dequant(tensor) -> np.ndarray:
    """把 GGUF 张量反量化为 fp32 的 (out, in) 逻辑布局."""
    raw = np.ascontiguousarray(tensor.data)
    t = tensor.tensor_type
    if t == QT.BF16:
        u = raw.view(np.uint16).astype(np.uint32) << 16
        mat = u.view(np.float32)
    elif t == QT.F16:
        mat = raw.view(np.float16).astype(np.float32)
    elif t == QT.F32:
        mat = raw.view(np.float32)
    else:
        from gguf import quants
        mat = quants.dequantize(raw, t)
    # GGUF 存储为 (ne0, ne1, ...). 标量维 squeeze 后即逻辑行主序.
    shape = tuple(int(x) for x in reversed(tensor.shape.tolist()))
    return mat.reshape(shape)


class GGUFModel:
    """只读 GGUF 权重容器, 按名字取 fp32 张量."""

    def __init__(self, path: str | Path):
        if not available():
            raise RuntimeError("gguf 包未安装")
        self.path = str(path)
        self._reader = GGUFReader(self.path)
        self.meta: dict[str, object] = {}
        for k, f in self._reader.fields.items():
            try:
                v = f.contents()
                if hasattr(v, "tolist"):
                    v = v.tolist()
            except Exception:
                v = f.parts[-1]
            self.meta[k] = v
        self._names = [t.name for t in self._reader.tensors]
        self._by_name = {t.name: t for t in self._reader.tensors}
        self._cache: dict[str, np.ndarray] = {}

    def tensor(self, name: str) -> np.ndarray:
        if name not in self._by_name and (name + ".weight") in self._by_name:
            name = name + ".weight"
        if name not in self._by_name:
            raise KeyError(name)
        if name not in self._cache:
            self._cache[name] = _dequant(self._by_name[name])
        return self._cache[name]

    def has(self, name: str) -> bool:
        return name in self._by_name

    def names(self) -> list[str]:
        return list(self._names)

    def get_int(self, key: str, default: int = 0) -> int:
        v = self.meta.get(key, default)
        return int(v) if v is not None else default

    def get_float(self, key: str, default: float = 0.0) -> float:
        v = self.meta.get(key, default)
        return float(v) if v is not None else default

    def tokens(self) -> list[bytes]:
        """RWKV 词表在 GGUF 里以 `\\xNN` 转义字面量存字符串, 需反转义为字节."""
        import codecs
        out = []
        for t in self.meta.get("tokenizer.ggml.tokens", []):
            if not isinstance(t, str):
                out.append(bytes(t))
                continue
            try:
                out.append(codecs.escape_decode(t.encode("latin-1"))[0])
            except Exception:
                out.append(t.encode("utf-8"))
        return out