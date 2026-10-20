"""约束筛选: 在同一训练协议下比较若干结构性改动对验证 ppl 的影响.

每个变体单独进程运行, 避免同进程累积构建导致的耗时/内存失真.
"""
from __future__ import annotations

import subprocess
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sunnynet.init import orthogonal_spectrum_  # noqa: E402
from sunnynet.model import SunnyConfig, SunnyNet  # noqa: E402
from sunnynet.ops import CenteredScaleNorm, softwedge  # noqa: E402

VARIANTS = ("base", "fastinit", "statenorm", "innergate", "residual",
            "resid_norm", "nl", "resid_nl", "resid_scale", "resid_mul",
            "hmax256", "hmax512", "hmax1024", "inner256", "rank128",
            "ln", "ln_resid")

_CACHE = {}


class _Names:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def _emit_gated(self, u, states, s_in):
    h = self.W_z(u)
    g = torch.sigmoid(self.W_g(u))
    y = self.norm_mix(softwedge(h, self.a.tanh()) * g)
    return self.W_c(y) + self.b_gain * states + self.W_s(self._feedback(states, s_in))


def _apply_variant(m: SunnyNet, variant: str):
    if variant.startswith("smin"):
        s = float(variant[4:]) / 1000.0
        for b in m.blocks:
            b.norm_in.s_min = s
            b.norm_mix.s_min = s
        m.norm_out.s_min = s
    if variant in ("ln", "ln_resid"):
        # 对照组: 用 LayerNorm 替换 CenteredScaleNorm (同行全部使用 LN)
        for b in m.blocks:
            b.norm_in = nn.LayerNorm(b.dim)
            b.norm_mix = nn.LayerNorm(b.inner)
        m.norm_out = nn.LayerNorm(m.cfg.dim)
    if variant in ("fastinit", "norm_fast"):
        frac = torch.log(torch.tensor(256.0 / 2.0)) / torch.log(torch.tensor(2048.0 / 2.0))
        for b in m.blocks:
            s = frac * (torch.arange(b.dim, dtype=torch.float32) + 0.5) / b.dim
            theta = torch.log(s) - torch.log1p(-s)
            with torch.no_grad():
                b.theta.copy_(theta)
    if variant == "upscale_state":
        for b in m.blocks:
            with torch.no_grad():
                b.b_gain.fill_(4.0)
    if variant in ("nl", "resid_nl"):
        import math
        for b in m.blocks:
            with torch.no_grad():
                # a 通过 tanh 映射, 反解使 a_eff=0.999 (非线性最强)
                b.a.fill_(math.atanh(0.999))
    if variant == "resid_scale":
        for b in m.blocks:
            with torch.no_grad():
                b.a.fill_(1.0)
    if variant == "innergate":
        for b in m.blocks:
            b.W_g = nn.Linear(b.dim, b.inner, bias=False)
            with torch.no_grad():
                orthogonal_spectrum_(b.W_g.weight, 1.0)
            b._emit = types.MethodType(_emit_gated, b)


def variant_model(variant: str, vocab: int, seed: int) -> SunnyNet:
    torch.manual_seed(seed)
    cfg = SunnyConfig.small(vocab)
    cfg.scan_chunk = 64
    if variant == "innergate":
        cfg.inner = 128
    if variant == "hmax256":
        cfg.h_max = 256.0
    if variant == "hmax512":
        cfg.h_max = 512.0
    if variant == "hmax1024":
        cfg.h_max = 1024.0
    if variant == "inner256":
        cfg.inner = 256
    if variant == "rank128":
        cfg.rank = 128
    ab = {}
    if variant in ("statenorm", "norm_fast", "resid_norm"):
        ab["state_norm"] = True
    if variant in ("residual", "resid_norm", "resid_nl", "resid_scale", "resid_mul",
                   "hmax256", "hmax512", "hmax1024", "inner256", "rank128",
                   "ln_resid") or variant.startswith("smin") or variant.startswith("rel"):
        ab["residual"] = True
    if variant == "resid_mul":
        ab["state_mul"] = True
    if ab:
        cfg.ablation = ab
    m = SunnyNet(cfg)
    _apply_variant(m, variant)
    return m


def _stage_messages():
    return [
        "加载真实语料",
        "按变体训练 300 步",
        "统计状态通路占比",
    ]


def _cell(variant: str, steps: int, threads: int, seed: int) -> dict:
    import math

    import numpy as np

    from train.train import DEFAULT_MIX, DomainMixer, evaluate, lr_at

    torch.set_num_threads(threads)
    from sunnynet.tokenizer import BPETokenizer

    tk = BPETokenizer.load(ROOT / "data" / "proc" / "tokenizer.json")
    vocab = tk.vocab_size

    model = variant_model(variant, vocab, seed)
    mixer = DomainMixer(DEFAULT_MIX, 256, 8, "train", seed=seed)
    vm = DomainMixer(DEFAULT_MIX, 256, 8, "val", seed=1)
    val_batches = vm.val_batches(8)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, betas=(0.9, 0.95),
                            weight_decay=0.01)
    best = float("inf")
    for step in range(steps):
        lr = lr_at(step, steps, 3e-3, 60)
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = mixer.batch()
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % 100 == 0 or step == steps - 1:
            vl = evaluate(model, val_batches)
            best = min(best, vl)

    # 状态通路占比: 直接看 W_c y / b*S / W_s fb 三项 RMS
    shares = []
    model.eval()
    with torch.no_grad():
        xb, _ = mixer.batch()
        for bi, blk in enumerate(model.blocks):
            buf = {}
            orig = blk._emit

            def probe(u, states, s_in, _b=blk, _buf=buf, _o=orig):
                y = _b.norm_mix(softwedge(_b.W_z(u), _b.a.tanh()))
                sn = _b.norm_s(states) if _b.norm_s is not None else states
                _buf["t1"] = _b.W_c(y).pow(2).mean().sqrt().item()
                _buf["t2"] = (_b.b_gain * sn).pow(2).mean().sqrt().item()
                _buf["t3"] = _b.W_s(_b._feedback(states, s_in)).pow(2).mean().sqrt().item()
                return _o(u, states, s_in)

            blk._emit = probe
            inp = model.embed(xb) if bi == 0 else cur
            cur, _, _ = blk(inp, torch.zeros(8, blk.dim), torch.zeros(8, blk.taps - 1, blk.dim))
            blk._emit = orig
            tot = buf["t1"] + buf["t2"] + buf["t3"]
            shares.append({"layer": bi, "Wc_y": round(buf["t1"], 5),
                           "bS": round(buf["t2"], 5), "Ws_fb": round(buf["t3"], 5),
                           "state_share": round((buf["t2"] + buf["t3"]) / max(tot, 1e-12), 4)})

    return {"variant": variant, "seed": seed, "steps": steps,
            "best_val": round(best, 6), "ppl": round(math.exp(min(best, 20)), 4),
            "params": model.num_params(non_embedding=True), "shares": shares}


if __name__ == "__main__":
    if sys.argv[1] == "--cell":
        import json
        print("RESULT " + json.dumps(_cell(sys.argv[2], int(sys.argv[3]),
                                          int(sys.argv[4]), int(sys.argv[5]))))
        raise SystemExit(0)

    import argparse
    import json

    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default=str(ROOT / "experiments" / "results" / "screen.json"))
    a = ap.parse_args()

    out = []
    for v in a.variants.split(","):
        r = subprocess.run(
            [sys.executable, "-X", "utf8", str(Path(__file__).resolve()),
             "--cell", v, str(a.steps), str(a.threads), str(a.seed)],
            capture_output=True, text=True, cwd=str(ROOT))
        line = [l for l in r.stdout.splitlines() if l.startswith("RESULT")]
        if not line:
            print(f"{v}: 失败\n{r.stdout[-800:]}\n{r.stderr[-800:]}")
            continue
        d = json.loads(line[-1][7:])
        out.append(d)
        sh = sum(s["state_share"] for s in d["shares"]) / len(d["shares"])
        print(f"{d['variant']:10s} ppl={d['ppl']:8.3f}  stack={d['params']:7d}  "
              f"状态通路占比={sh * 100:5.1f}%")

    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print(f"-> {a.out}")