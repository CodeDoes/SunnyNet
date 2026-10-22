"""把训练 checkpoint 打包成可发布的权重资产.

产物:
  <name>.safetensors   HF / transformers 兼容 (config 与训练摘要在 metadata)
  <name>.pt            原始 checkpoint (可直接被 sunnynet-cli --ckpt 读取)
  <name>.runtime.bin   C Runtime 打包 fp32 布局 (零依赖推理)
  <name>.card.json     模型卡 (架构超参、训练摘要、sha256 清单)

INT4 目前只在 RWKV-7 侧实现 (<name>.q4.bin 由 _export_rwkv7.py 生成),
SunnyNet 的 runtime blob 尚无对应量化格式, 故不在此导出。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
from pathlib import Path

import torch

from sunnynet.cli import load_model

__all__ = ["sha256_of", "export_safetensors", "export_runtime_blob",
           "write_model_card", "export_all", "main"]

MAGIC = 0x4E4E5355

# Runtime 内核只实现了默认路径, 其余消融组合必须拒绝导出
ALLOWED_ABLATION = dict(decay_gate=True, circulant=True, local_taps=True,
                        current=True, residual=True, state_norm=False,
                        state_mul=False)


def sha256_of(path: str | Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _check_ablation(cfg) -> None:
    ab = cfg.ablation or {}
    for k, want in ALLOWED_ABLATION.items():
        if ab.get(k, want) != want:
            raise SystemExit(f"ablation[{k}]={ab.get(k)}; Runtime 只支持默认路径")


def _ws_eff(blk) -> torch.Tensor:
    W = blk.W_s.weight
    t0, t1, t2 = blk.circ.t[0], blk.circ.t[1], blk.circ.t[2]
    return t0 * torch.roll(W, -1, dims=1) + t1 * W + t2 * torch.roll(W, 1, dims=1)


def export_safetensors(ckpt: str | Path, out: str | Path) -> dict:
    """导出 HF 风格 safetensors (权重 + config 写进 metadata).

    绑定嵌入时 `head.weight` 与 `embed.weight` 共享存储, safetensors 拒绝
    重复保存同一块内存; 这里把 head 摘掉并在 metadata 标注, 加载方按
    `tie_embeddings` 直接复用 embed 即可。
    """
    from safetensors.torch import save_file

    m, cfg, summary = load_model(str(ckpt))
    m.eval()
    sd = {k: v.detach().float().contiguous().cpu()
          for k, v in m.state_dict().items()}
    tied = cfg.tie_embeddings and "head.weight" in sd and "embed.weight" in sd
    if tied:
        sd.pop("head.weight")
    meta = {
        "format": "pt",
        "architecture": "SunnyNet",
        "sunnynet.config": json.dumps(cfg.__dict__, ensure_ascii=False),
        "sunnynet.summary": json.dumps(summary, ensure_ascii=False),
        "sunnynet.version": "1",
        "sunnynet.tied_head": "1" if tied else "0",
    }
    outp = Path(out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    save_file(sd, str(outp), metadata=meta)
    return {"file": str(outp), "bytes": outp.stat().st_size,
            "tensors": len(sd), "params": m.num_params(),
            "params_non_embed": m.num_params(non_embedding=True),
            "tied_head": tied}


def export_runtime_blob(ckpt: str | Path, out: str | Path) -> dict:
    """导出 C Runtime 的整块 fp32 布局 (头部 10 个 int32, 见 sn_load)."""
    m, cfg, _ = load_model(str(ckpt))
    m.eval()
    _check_ablation(cfg)
    if not cfg.tie_embeddings:
        raise SystemExit("Runtime 只支持绑定嵌入 (tie_embeddings=True)")

    D, I, R, K, L = cfg.dim, cfg.inner, cfg.rank, cfg.taps, cfg.layers
    eps = 0.0
    for b in m.blocks:
        nm = b.norm_in
        eps = max(eps, nm.s_min * nm.s_min + nm.eps)
    needs_clamp = any(b._need_clamp for b in m.blocks)
    p_floor = 1.0 / cfg.h_max

    parts = [struct.pack("<10i", MAGIC, cfg.vocab_size, D, I, R, K, L, 8, 1,
                         int(needs_clamp))]

    def put(t):
        parts.append(t.detach().float().contiguous().view(-1).numpy().tobytes())

    for b in m.blocks:
        put(b.norm_in.gamma)
        put(b.norm_in.center)
        put(b.norm_mix.gamma)
        put(b.norm_mix.center)
        put(b.dw.weight.detach().reshape(D, K).transpose(0, 1).contiguous())
        put(b.V.weight.transpose(0, 1).contiguous())
        put(b.U.weight.transpose(0, 1).contiguous())
        put(b.W_z.weight.transpose(0, 1).contiguous())
        put(b.W_c.weight.transpose(0, 1).contiguous())
        put(_ws_eff(b).transpose(0, 1).contiguous())
        put(b.delta().detach())
        parts.append(struct.pack("<2f", float(torch.tanh(b.a.detach())),
                                 float(b.b_gain.detach())))
    parts.append(struct.pack("<2f", eps, p_floor))
    put(m.norm_out.gamma)
    put(m.norm_out.center)
    put(m.embed.weight)
    put(m.embed.weight.transpose(0, 1).contiguous())

    buf = b"".join(parts)
    outp = Path(out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_bytes(buf)
    return {"file": str(outp), "bytes": len(buf), "params": m.num_params(),
            "dim": D, "inner": I, "rank": R, "taps": K, "layers": L,
            "vocab": cfg.vocab_size, "needs_clamp": needs_clamp}


def write_model_card(out_path: str | Path, card: dict) -> Path:
    outp = Path(out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(card, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    return outp


def export_all(ckpt: str | Path, out_dir: str | Path, name: str) -> dict:
    """一次导出全部产物并返回带 sha256 的清单."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    arts = {}
    pt_copy = out_dir / f"{name}.pt"
    if Path(ckpt).resolve() != pt_copy.resolve():
        shutil.copy2(ckpt, pt_copy)
    arts["checkpoint"] = {"file": str(pt_copy), "bytes": pt_copy.stat().st_size}

    arts["safetensors"] = export_safetensors(ckpt, out_dir / f"{name}.safetensors")
    arts["runtime_fp32"] = export_runtime_blob(ckpt,
                                               out_dir / f"{name}.runtime.bin")

    for v in arts.values():
        f = v.get("file")
        if f and Path(f).exists():
            v["sha256"] = sha256_of(f)

    _, cfg, summary = load_model(str(ckpt))
    card = {
        "name": name,
        "architecture": "SunnyNet",
        "publisher": "sunny-cx",
        "license": "apache-2.0",
        "config": cfg.__dict__,
        "training": summary,
        "artifacts": arts,
        "usage": {
            "pytorch": f"python -m sunnynet.cli --ckpt {name}.pt --prompt '子曰：'",
            "c_runtime": (f"python -m sunnynet.cli --ckpt {name}.pt "
                          f"--backend c --blob {name}.runtime.bin"),
            "openai_api": "sunnynet-serve --ckpt <path> --port 8080",
        },
    }
    card_path = write_model_card(out_dir / f"{name}.card.json", card)
    card["card"] = str(card_path)
    return card


def main(argv=None):
    ap = argparse.ArgumentParser(prog="sunnynet-release",
                                 description="导出可发布权重资产")
    ap.add_argument("ckpt", help="训练产出的 checkpoint (.pt)")
    ap.add_argument("--out-dir", default="dist/release")
    ap.add_argument("--name", default=None, help="资产名, 默认取 ckpt 文件名")
    args = ap.parse_args(argv)

    name = args.name or Path(args.ckpt).stem
    card = export_all(args.ckpt, args.out_dir, name)
    print(json.dumps(card, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())