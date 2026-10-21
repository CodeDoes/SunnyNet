"""逐元素算子热点探测: 成对竞争 / fused 激活 / inplace."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]


def _bench(fn, warmup=3, iters=20) -> float:
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters


def cell_compete(rows: int, dim: int, mode: str) -> dict:
    """组内成对竞争: 朴素 O(d^2) vs 三种改写."""
    torch.manual_seed(0)
    g = torch.randn(rows, dim)
    d = dim

    def naive():
        tot = torch.zeros_like(g)
        for j in range(d):
            for k in range(d):
                if j != k:
                    tot += torch.sigmoid(g - g[:, j:j + 1]) * torch.sigmoid(g - g[:, k:k + 1])
        return tot

    def shifted():
        # 复用已算的 sigmoid(g_i - g_j), 转置取负关系
        tot = torch.zeros_like(g)
        for j in range(d):
            col = torch.sigmoid(g - g[:, j:j + 1])
            tot += col.sum(dim=-1, keepdim=True) - col.abs() * 0 + col
        return tot

    def matrix():
        # 一次算出 d*d 的 sigmoid 差矩阵, 再沿两轴求和
        diff = g.unsqueeze(-1) - g.unsqueeze(-2)
        s = torch.sigmoid(diff)
        ssum = s.sum(dim=-1) + s.sum(dim=-2)
        return ssum

    fn = {"naive": naive, "shifted": shifted, "matrix": matrix}[mode]
    t = _bench(fn)
    return {"cell": f"compete_{mode}_{rows}x{dim}", "seconds": t}


def cell_act(rows: int, dim: int, mode: str) -> dict:
    """同一数学运算的不同写法, 看是否有 fused kernel 可用."""
    x = torch.randn(rows, dim)

    def manual_silu():
        return x * torch.sigmoid(x)

    def f_silu():
        return F.silu(x)

    def manual_sigmoid():
        return 1.0 / (1.0 + torch.exp(-x))

    def t_sigmoid():
        return torch.sigmoid(x)

    def manual_swish_hard():
        return F.hardswish(x)

    fn = {"manual_silu": manual_silu, "f_silu": f_silu,
          "manual_sigmoid": manual_sigmoid, "t_sigmoid": t_sigmoid,
          "hardswish": manual_swish_hard}[mode]
    t = _bench(fn)
    return {"cell": f"act_{mode}_{rows}x{dim}", "seconds": t}


def cell_inplace(rows: int, dim: int, layers: int, mode: str) -> dict:
    """算子链: out-of-place (每步新分配) vs inplace."""
    x = torch.randn(rows, dim)

    def oop():
        h = x
        for _ in range(layers):
            h = torch.sigmoid(h)
            h = h * (1.0 - h)
        return h

    def inp():
        h = x.clone()
        for _ in range(layers):
            h.sigmoid_()
            h.mul_(1.0 - h)
        return h

    fn = {"oop": oop, "inplace": inp}[mode]
    t = _bench(fn)
    return {"cell": f"inplace_{mode}_{rows}x{dim}xL{layers}", "seconds": t}


def cell_addcmul(rows: int, dim: int, mode: str) -> dict:
    """融合点算子 addcmul_ 是否省调度开销."""
    a = torch.randn(rows, dim)
    b = torch.randn(rows, dim)
    g = torch.sigmoid(torch.randn(rows, dim))

    def separate():
        return a * (1.0 - g) + g * b

    def fused():
        return torch.addcmul(a, g, b - a)

    fn = {"separate": separate, "fused": fused}[mode]
    t = _bench(fn)
    return {"cell": f"addcmul_{mode}_{rows}x{dim}", "seconds": t}


CELLS = {
    "compete": cell_compete,
    "act": cell_act,
    "inplace": cell_inplace,
    "addcmul": cell_addcmul,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default="")
    args = ap.parse_args()
    if args.cell:
        name, kw = args.cell.split(":", 1)
        print(json.dumps(CELLS[name](**json.loads(kw))))
        return
    print(json.dumps({"hint": "use --cell name:{json}"}))


if __name__ == "__main__":
    main()