"""RWKV 官方词表文件 (rwkv_vocab_v20230424.txt) 的读写.

行格式: `<id> '<转义字节>' <字节数>`。token 是原始字节, 编码走贪心最长匹配。
不依赖 GGUF, 供 FLA / GGUF 两条权重路线共用。
"""
from __future__ import annotations

__all__ = ["RWKVVocab"]


class RWKVVocab:
    """从官方词表文件构造的字节级贪心最长匹配分词器."""

    def __init__(self, path: str):
        self.idx2tok: dict[int, bytes] = {}
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                idx, _, rest = line.partition(" ")
                try:
                    tid = int(idx)
                except ValueError:
                    continue
                payload = rest.rstrip()
                sp = payload.rfind(" ")
                if sp > 0:
                    payload = payload[:sp]
                if len(payload) >= 2 and payload[0] == "'" and payload[-1] == "'":
                    payload = payload[1:-1]
                raw = self._unescape(payload)
                self.idx2tok[tid] = raw
        self._build_trie()

    @staticmethod
    def _unescape(s: str) -> bytes:
        """词表用 `\\xNN` / `\\t` / `\\'` 等转义写字面字节, 手工还原避免告警。"""
        out = bytearray()
        i, n = 0, len(s)
        while i < n:
            c = s[i]
            if c == "\\" and i + 1 < n:
                e = s[i + 1]
                if e == "x" and i + 3 < n:
                    try:
                        out.append(int(s[i + 2:i + 4], 16))
                        i += 4
                        continue
                    except ValueError:
                        pass
                m = {"t": 9, "n": 10, "r": 13, "0": 0, "\\": 92,
                     "'": 39, '"': 34}
                if e in m:
                    out.append(m[e])
                    i += 2
                    continue
            out.extend(c.encode("latin-1", errors="replace"))
            i += 1
        return bytes(out)

    def _build_trie(self):
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