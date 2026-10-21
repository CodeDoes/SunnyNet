"""C 整模型前向 vs torch forward_step 的逐 token 数值比对.

对同一批 token 逐步推进, 比对每一步 logits 与最终 argmax。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sunnynet.cli import load_model  # noqa: E402
from sunnynet.runtime.model import SnModel  # noqa: E402

CKPT = ROOT / "experiments/ckpts_ood/sunnynet.pt"
BLOB = ROOT / "experiments/ckpts_ood/sunnynet.runtime.bin"


def main():
    torch.set_num_threads(1)
    m, cfg, _ = load_model(str(CKPT))
    m.eval()
    blob = BLOB
    if len(sys.argv) > 1:
        blob = Path(sys.argv[1])
    rt = SnModel(blob, cfg.vocab_size, cfg.dim, cfg.inner, cfg.rank,
                 cfg.taps, cfg.layers, 8, int(cfg.tie_embeddings))

    torch.manual_seed(7)
    toks = torch.randint(0, cfg.vocab_size, (48,)).tolist()

    s = m.init_state(1)
    diffs = []
    argmax_match = 0
    with torch.no_grad():
        for i, t in enumerate(toks):
            lg_t, s = m.forward_step(torch.tensor([t]), s)
            lg_c = rt.step(t)
            a = lg_t[0].numpy()
            b = torch.tensor(lg_c).numpy()
            d = float(abs(a - b).max())
            diffs.append(d)
            if int(a.argmax()) == int(b.argmax()):
                argmax_match += 1

    # 预填路径一致性
    rt2 = SnModel(blob, cfg.vocab_size, cfg.dim, cfg.inner, cfg.rank,
                  cfg.taps, cfg.layers, 8, int(cfg.tie_embeddings))
    lg_pref = rt2.prefill(toks)
    lg_last_ref = None
    s = m.init_state(1)
    with torch.no_grad():
        lg, s = m(torch.tensor([toks]), return_state=True)
        lg_last_ref = lg[0, -1].numpy()
    d_pref = float(abs(torch.tensor(lg_pref).numpy() - lg_last_ref).max())

    out = {
        "tokens": len(toks),
        "step_maxdiff": max(diffs),
        "step_meandiff": sum(diffs) / len(diffs),
        "argmax_match": f"{argmax_match}/{len(toks)}",
        "prefill_last_maxdiff": d_pref,
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())