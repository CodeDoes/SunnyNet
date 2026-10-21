"""sunnynet-cli: 加载 SunnyNet 权重, 在 CPU 上自回归生成文本.

    python -m sunnynet.cli --ckpt runs/small.pt --prompt "子曰：" --max-new 80

挂点: 权重加载 + 分词器 + 自回归解码循环。递推状态走 SunnyNet 的
forward_step (与整段前向逐位一致), --runtime 可把单步递推交给独立 C 内核。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from .model import SunnyConfig, SunnyNet
from .tokenizer import BPETokenizer

__all__ = ["main", "load_model", "decode_stream"]

REPO = Path(__file__).resolve().parent.parent
TOKENIZER = REPO / "data" / "proc" / "tokenizer.json"


def load_model(ckpt: str, arch: str | None = None):
    d = torch.load(ckpt, map_location="cpu", weights_only=False)
    a = arch or d.get("arch", "sunnynet")
    if a != "sunnynet":
        raise SystemExit(f"本 CLI 只跑 sunnynet 权重, 收到 arch={a}")
    cfg = SunnyConfig(**d["config"])
    m = SunnyNet(cfg)
    m.load_state_dict(d["model"])
    m.eval()
    return m, cfg, d.get("summary", {})


def _sample(last: torch.Tensor, temperature: float, top_k: int, top_p: float):
    if temperature <= 0:
        return int(last.argmax())
    last = last / temperature
    if top_k > 0:
        v, _ = torch.topk(last, min(top_k, last.numel()))
        last = last.masked_fill(last < v[-1], float("-inf"))
    if 0.0 < top_p < 1.0:
        srt, idx = torch.sort(last, descending=True)
        cum = torch.softmax(srt, dim=-1).cumsum(-1)
        cut = cum > top_p
        cut[1:] = cut[:-1].clone()
        cut[0] = False
        srt = srt.masked_fill(cut, float("-inf"))
        last = torch.full_like(last, float("-inf")).scatter(0, idx, srt)
    return int(torch.multinomial(torch.softmax(last, dim=-1), 1))


@torch.no_grad()
def decode_stream(model, tk, prompt: str, max_new: int = 80,
                  temperature: float = 0.8, top_k: int = 40, top_p: float = 0.95,
                  seed: int | None = None, verbose: bool = False,
                  runtime=None):
    """逐 token 生成, yield 新 token 的文本片段; 返回 (全文, 统计).

    runtime 给定时, 解码走独立 C 内核 (整模型单步), 采样与分词仍在 torch。
    """
    if seed is not None:
        torch.manual_seed(seed)
    ids = tk.encode(prompt) or [tk.bos_id]
    t0 = time.perf_counter()
    if runtime is not None:
        logits = torch.tensor(runtime.prefill(ids))
    else:
        s = model.init_state(1)
        logits, s = model(torch.tensor([ids], dtype=torch.long), return_state=True)
        logits = logits[0, -1]
    prefill_s = time.perf_counter() - t0
    out = list(ids)
    last = logits
    step_times = []
    for _ in range(max_new):
        nxt = _sample(last, temperature, top_k, top_p)
        out.append(nxt)
        piece = tk.decode([nxt])
        yield piece
        if nxt == tk.eos_id:
            break
        t0 = time.perf_counter()
        if runtime is not None:
            last = torch.tensor(runtime.step(nxt))
        else:
            last, s = model.forward_step(torch.tensor([nxt]), s)
            last = last[0]
        step_times.append(time.perf_counter() - t0)
    stats = {
        "prefill_tokens": len(ids),
        "prefill_ms": prefill_s * 1e3,
        "backend": "c" if runtime is not None else "torch",
        "gen_tokens": len(step_times),
        "decode_ms_median": (sorted(step_times)[len(step_times) // 2] * 1e3
                             if step_times else 0.0),
        "decode_ms_per_token": (sum(step_times) / len(step_times) * 1e3
                                if step_times else 0.0),
    }
    decode_stream.stats = stats
    decode_stream.text = tk.decode(out)
    decode_stream.state = None


def _main_gguf(args) -> int:
    """外部 GGUF 权重路径 (RWKV-7), 不经 PyTorch."""
    from .runtime.rwkv7 import RWKV7
    t0 = time.perf_counter()
    m = RWKV7(args.gguf)
    load_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    text, ids = m.generate(args.prompt, max_new=args.max_new,
                           temperature=args.temperature, top_k=args.top_k,
                           seed=args.seed)
    gen_s = time.perf_counter() - t0
    st = {
        "backend": "gguf/rwkv7",
        "load_ms": load_s * 1e3,
        "gen_tokens": len(ids),
        "decode_ms_per_token": gen_s / max(len(ids), 1) * 1e3,
    }
    if args.json:
        print(json.dumps({"prompt": args.prompt, "text": text,
                          "arch": "rwkv7", "dim": m.n_embd, "layers": m.n_layer,
                          "stats": st}, ensure_ascii=False, indent=2))
    else:
        sys.stdout.write(text)
        print()
        if args.stats:
            print(f"[rwkv7 {m.n_layer}L d{m.n_embd} vocab {m.vocab} "
                  f"backend gguf/numpy]")
            print(f"[load {st['load_ms']:.1f} ms; {st['gen_tokens']} tok "
                  f"{st['decode_ms_per_token']:.2f} ms/tok (numpy 路径)]")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="sunnynet-cli",
                                 description="SunnyNet CPU 推理")
    ap.add_argument("--ckpt", default=None, help="SunnyNet checkpoint 路径")
    ap.add_argument("--gguf", default=None,
                    help="外部 GGUF 权重路径 (目前支持 RWKV-7), 与 --ckpt 二选一")
    ap.add_argument("--prompt", default="", help="提示词")
    ap.add_argument("--max-new", type=int, default=80)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--tokenizer", default=str(TOKENIZER))
    ap.add_argument("--threads", type=int, default=0, help="0 = 不限制")
    ap.add_argument("--backend", choices=["torch", "c"], default="torch",
                    help="c = 纯 C 整模型单步 (需先导出 .runtime.bin)")
    ap.add_argument("--blob", default=None, help="--backend c 的权重 blob 路径")
    ap.add_argument("--stats", action="store_true", help="输出解码耗时统计")
    ap.add_argument("--json", action="store_true", help="以 JSON 输出结果")
    args = ap.parse_args(argv)

    if args.gguf:
        return _main_gguf(args)
    if not args.ckpt:
        raise SystemExit("需要 --ckpt 或 --gguf")

    if args.threads > 0:
        torch.set_num_threads(args.threads)

    tkp = Path(args.tokenizer)
    if not tkp.exists():
        raise SystemExit(f"分词器不存在: {tkp}")
    tk = BPETokenizer.load(str(tkp))
    model, cfg, summary = load_model(args.ckpt)
    n = model.num_params()

    runtime = None
    if args.backend == "c":
        from .runtime.model import SnModel
        blob = Path(args.blob) if args.blob else Path(args.ckpt).with_suffix(".runtime.bin")
        if not blob.exists():
            raise SystemExit(f"缺少权重 blob: {blob}\n"
                             f"先运行: python -X utf8 experiments/_export_runtime.py "
                             f"{args.ckpt} {blob}")
        runtime = SnModel(blob, cfg.vocab_size, cfg.dim, cfg.inner, cfg.rank,
                          cfg.taps, cfg.layers, 8, int(cfg.tie_embeddings))

    pieces = []
    for p in decode_stream(model, tk, args.prompt, args.max_new,
                           args.temperature, args.top_k, args.top_p,
                           args.seed, verbose=True, runtime=runtime):
        pieces.append(p)
        sys.stdout.write(p)
        sys.stdout.flush()
    st = getattr(decode_stream, "stats", {})

    if args.json:
        print()
        print(json.dumps({"prompt": args.prompt, "text": "".join(pieces),
                          "params": n, "dim": cfg.dim, "layers": cfg.layers,
                          "stats": st}, ensure_ascii=False, indent=2))
    elif args.stats:
        print()
        print(f"[params {n/1e6:.2f}M dim {cfg.dim} layers {cfg.layers} "
              f"backend {st.get('backend', 'torch')}]")
        print(f"[prefill {st['prefill_tokens']} tok {st['prefill_ms']:.1f} ms; "
              f"decode {st['gen_tokens']} tok "
              f"{st['decode_ms_per_token']:.2f} ms/tok]")
    else:
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())