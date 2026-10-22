"""权重发布导出契约测试 (不联网, 不上传)."""
from __future__ import annotations

import json
import struct

import pytest
import torch

from sunnynet import release
from sunnynet.model import SunnyConfig, SunnyNet


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    """造一个极小但结构完整的 SunnyNet checkpoint."""
    d = tmp_path_factory.mktemp("rel")
    cfg = SunnyConfig(vocab_size=256, dim=32, inner=48, rank=16, layers=2,
                      h_max=2048.0)
    m = SunnyNet(cfg)
    p = d / "tiny.pt"
    torch.save({"arch": "sunnynet", "model": m.state_dict(),
                "config": cfg.__dict__,
                "summary": {"best_ppl": 12.5, "steps_done": 7}}, p)
    return p


def test_sha256_of_matches_hashlib(tmp_path):
    import hashlib
    f = tmp_path / "x.bin"
    f.write_bytes(b"hello sunnynet")
    assert release.sha256_of(f) == hashlib.sha256(b"hello sunnynet").hexdigest()


def test_export_safetensors_roundtrip(ckpt, tmp_path):
    from safetensors import safe_open
    info = release.export_safetensors(ckpt, tmp_path / "m.safetensors")
    assert info["tensors"] > 0 and info["params"] > 0
    assert info["tied_head"] is True
    with safe_open(str(tmp_path / "m.safetensors"), framework="pt") as f:
        meta = f.metadata()
        assert meta["architecture"] == "SunnyNet"
        assert meta["sunnynet.tied_head"] == "1"
        cfg = json.loads(meta["sunnynet.config"])
        assert cfg["vocab_size"] == 256 and cfg["layers"] == 2
        keys = set(f.keys())
        # 绑定嵌入: head 与 embed 共享存储, safetensors 拒绝重复保存
        assert "head.weight" not in keys
        assert "embed.weight" in keys


def test_export_safetensors_all_tensors_finite(ckpt, tmp_path):
    from safetensors import safe_open
    release.export_safetensors(ckpt, tmp_path / "m2.safetensors")
    with safe_open(str(tmp_path / "m2.safetensors"), framework="pt") as f:
        for k in f.keys():
            assert torch.isfinite(f.get_tensor(k)).all(), k


def test_export_runtime_blob_header(ckpt, tmp_path):
    info = release.export_runtime_blob(ckpt, tmp_path / "m.runtime.bin")
    raw = (tmp_path / "m.runtime.bin").read_bytes()
    hdr = struct.unpack("<10i", raw[:40])
    assert hdr[0] == release.MAGIC
    assert hdr[1] == 256 and hdr[2] == 32 and hdr[3] == 48 and hdr[4] == 16
    assert hdr[6] == 2, "层数"
    assert hdr[7] == 8, "分组数"
    assert info["bytes"] == len(raw)


def test_export_runtime_blob_rejects_bad_ablation(tmp_path, ckpt):
    d = torch.load(ckpt, map_location="cpu", weights_only=False)
    d["config"] = dict(d["config"], ablation={"residual": False})
    p = tmp_path / "bad.pt"
    torch.save(d, p)
    with pytest.raises(SystemExit, match="ablation"):
        release.export_runtime_blob(p, tmp_path / "o.bin")


def test_export_all_produces_manifest_with_hashes(ckpt, tmp_path):
    card = release.export_all(ckpt, tmp_path / "out", "tiny")
    assert set(card["artifacts"]) == {"checkpoint", "safetensors", "runtime_fp32"}
    for k, v in card["artifacts"].items():
        assert v["sha256"] and len(v["sha256"]) == 64, k
        assert v["bytes"] > 0
    assert card["license"] == "apache-2.0"
    assert card["training"]["best_ppl"] == 12.5
    # 卡片自身落盘且可解析
    saved = json.loads((tmp_path / "out" / "tiny.card.json").read_text("utf-8"))
    assert saved["name"] == "tiny"


def test_release_cli_writes_all(ckpt, tmp_path):
    rc = release.main([str(ckpt), "--out-dir", str(tmp_path / "cli"),
                       "--name", "cliart"])
    assert rc == 0
    out = tmp_path / "cli"
    for ext in (".pt", ".safetensors", ".runtime.bin", ".card.json"):
        assert (out / f"cliart{ext}").exists(), ext