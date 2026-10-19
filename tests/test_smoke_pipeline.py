"""端到端冒烟测试: 小语料 -> 分词 -> 训练 -> 评估, 全链路可跑通.

不依赖外网 (语料在测试内合成), 供 CI 验证架构改动没有破坏训练流程.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
DOMAINS = ("zh_ancient", "en_literature", "code_python")
SEED_TEXTS = {
    "zh_ancient": ["子曰学而时习之不亦说乎有朋自远方来不亦乐乎",
                   "道可道非常道名可名非常名", "天行健君子以自强不息"] * 200,
    "en_literature": ["It is a truth universally acknowledged that a single man "
                      "in possession of a good fortune must be in want of a wife"] * 200,
    "code_python": ["def f(x):\n    return x * 2 + 1\n\nfor i in range(10):\n    f(i)"] * 200,
}


@pytest.fixture(scope="module")
def tiny_corpus(tmp_path_factory):
    proc = tmp_path_factory.mktemp("proc")
    sys.path.insert(0, str(ROOT))
    from sunnynet.tokenizer import BPETokenizer

    texts = [t for v in SEED_TEXTS.values() for t in v]
    tk = BPETokenizer.train(texts, vocab_size=520, min_char_freq=1)
    tk.save(proc / "tokenizer.json")
    man = {"vocab_size": tk.vocab_size, "val_ratio": 0.1, "domains": {}}
    for d in DOMAINS:
        ids = []
        for t in SEED_TEXTS[d]:
            ids.extend(tk.encode(t))
        a = np.asarray(ids, dtype=np.uint16)
        cut = int(a.size * 0.9)
        a[:cut].tofile(proc / f"tokens_{d}.bin")
        a[cut:].tofile(proc / f"val_{d}.bin")
        man["domains"][d] = {"train_tokens": cut, "val_tokens": a.size - cut,
                             "train_files": 1, "val_files": 1}
    (proc / "manifest.json").write_text(json.dumps(man), encoding="utf-8")
    return proc


def _patch_proc_dir(monkeypatch, proc: Path):
    import build_corpus
    monkeypatch.setattr(build_corpus, "PROC", proc)


def test_train_smoke_both_archs(tiny_corpus, monkeypatch, tmp_path):
    _patch_proc_dir(monkeypatch, tiny_corpus)
    for arch in ("sunnynet", "transformer"):
        out = tmp_path / f"{arch}.json"
        code = (
            "import sys; sys.path.insert(0,r'%s');"
            "import build_corpus, pathlib;"
            "build_corpus.PROC=pathlib.Path(r'%s');"
            "import train.train as T;"
            "sys.argv=['x','--arch','%s','--steps','20','--batch-size','4',"
            "'--seq-len','64','--val-every','10','--val-batches','2',"
            "'--threads','2','--out',r'%s'];"
            "raise SystemExit(T.main())" % (ROOT, tiny_corpus, arch, out)
        )
        r = subprocess.run([PY, "-X", "utf8", "-c", code], cwd=str(ROOT),
                           capture_output=True, text=True)
        assert r.returncode == 0, f"{arch} 训练失败:\n{r.stdout[-2500:]}\n{r.stderr[-2500:]}"
        log = json.loads(out.read_text(encoding="utf-8"))
        assert log["summary"]["steps_done"] == 20
        assert log["summary"]["final_train_loss"] > 0
        assert log["params"] > 0


def test_evaluate_smoke(tiny_corpus, tmp_path, monkeypatch):
    """评估脚本需能在真实 checkpoint 上跑通并产出长距离探针结果."""
    import torch
    sys.path.insert(0, str(ROOT))
    from sunnynet.model import SunnyConfig, SunnyNet

    ck = tmp_path / "ck.pt"
    cfg = SunnyConfig(vocab_size=520, dim=64, inner=96, rank=32, layers=2)
    m = SunnyNet(cfg)
    torch.save({"model": m.state_dict(), "config": cfg.__dict__}, ck)

    code = (
        "import sys; sys.path.insert(0,r'%s');"
        "import build_corpus, pathlib;"
        "build_corpus.PROC=pathlib.Path(r'%s');"
        "import train.evaluate as E;"
        "sys.argv=['x','--ckpt',r'%s','--seq-len','64','--windows','2',"
        "'--recall-seqs','1','--skip-latency','--out',r'%s'];"
        "raise SystemExit(E.main())"
        % (ROOT, tiny_corpus, ck, tmp_path / "ev.json")
    )
    r = subprocess.run([PY, "-X", "utf8", "-c", code], cwd=str(ROOT),
                       capture_output=True, text=True)
    assert r.returncode == 0, f"评估失败:\n{r.stdout[-2500:]}\n{r.stderr[-2500:]}"
    res = json.loads((tmp_path / "ev.json").read_text(encoding="utf-8"))
    assert set(res["domain_ppl"]) == set(DOMAINS)
    assert "long_range" in res