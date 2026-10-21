"""Scaling 曲线: 等 token 预算下的参数阶梯, 判定 ppl 是否随规模单调下降.

背景: 批评称 "Scaling Law 失效". 但仓库此前只有 20M / 100M 两个点且 token 预算不同
(20M 200 步, 100M 100 步), 无法判定单调性. 本节补同协议阶梯.

协议: 固定 token 预算 (steps x batch x seq_len 对所有规模相同), lr 按规模分档
(大模型高 lr 会发散, §14.8 已实测), 单种子, 只报 trend 不报单点优劣.

用法:
  python -X utf8 experiments/_scaling_curve.py params              # 只报各档栈参数量
  python -X utf8 experiments/_scaling_curve.py train <name> [steps]
  python -X utf8 experiments/_scaling_curve.py run [steps]
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

# 规模阶梯: 同步放大 dim/inner/rank/layers, 保持架构族一致.
# 命名按栈参数量量级, 不按总参数(嵌入占 8192*dim, 不可忽略).
LADDER = {
    "s0":  (128, 192,  64,  4),     # ~0.37M, = small
    "s1":  (192, 320,  96,  6),
    "s2":  (288, 512, 160,  8),
    "s3":  (416, 768, 256, 10),
    "s4":  (640, 1280, 384, 6),     # ~16.8M, = proto20mw (减层加宽, §14.3)
    "s5":  (1024, 1792, 640, 14),   # ~93.8M, = p100m
}
# 大模型降 lr: §14.8 实测 100M 用 3e-3 对部分架构发散
LR = {"s0": 3e-3, "s1": 3e-3, "s2": 3e-3, "s3": 2e-3, "s4": 1.5e-3, "s5": 1e-3}


def params_report():
    code = (
        "import sys; sys.path.insert(0, r'" + str(ROOT) + "')\n"
        "from sunnynet.model import SunnyConfig, SunnyNet\n"
        "V=8192\n"
        "LAD=" + repr(LADDER) + "\n"
        "for n,(d,i,r,l) in LAD.items():\n"
        "    c=SunnyConfig(vocab_size=V,dim=d,inner=i,rank=r,layers=l,h_max=2048.0)\n"
        "    m=SunnyNet(c)\n"
        "    print('%-4s d=%4d i=%4d r=%4d L=%2d stack=%9d total=%9d'\n"
        "          %(n,d,i,r,l,m.num_params(non_embedding=True),m.num_params()))\n"
    )
    subprocess.run([PY, "-X", "utf8", "-c", code], cwd=str(ROOT))


def run_train(name: str, steps: int) -> Path:
    d, i, r, l = LADDER[name]
    out = ROOT / "experiments" / "results" / f"_scaling_{name}.json"
    ck = ROOT / "experiments" / "ckpts_scaling"
    ck.mkdir(parents=True, exist_ok=True)
    cmd = [PY, "-X", "utf8", str(ROOT / "train" / "train.py"),
           "--arch", "sunnynet", "--preset", "small", "--steps", str(steps),
           "--batch-size", "8", "--seq-len", "256", "--lr", str(LR[name]),
           "--warmup", "30", "--seed", "1234", "--threads", "8",
           "--val-every", str(max(1, steps // 2)), "--val-batches", "8",
           "--no-early-stop", "--out", str(out)]
    print(f"    cfg d={d} i={i} r={r} L={l} lr={LR[name]} steps={steps}")
    subprocess.run(cmd, cwd=str(ROOT), check=False)
    return out


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "params":
        params_report(); return
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        run_train(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 150); return

    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    names = sys.argv[2].split(",") if len(sys.argv) > 2 else list(LADDER)
    rows = []
    for n in names:
        print(f"--- scaling {n} steps={steps} ---")
        out = run_train(n, steps)
        if not out.exists():
            print("    无结果"); continue
        d = json.loads(out.read_text(encoding="utf-8"))
        s = d.get("summary", {})
        rows.append({"name": n, "stack": d.get("params_non_embed"),
                     "params": d.get("params"),
                     "ppl": s.get("best_ppl"), "val": s.get("best_val_loss"),
                     "diverged": s.get("diverged"),
                     "ms_per_token": s.get("ms_per_token_mean")})
        print(f"    stack={d.get('params_non_embed'):,} ppl={s.get('best_ppl')} "
              f"diverged={s.get('diverged')}")

    res = ROOT / "experiments" / "results" / "_scaling_curve.json"
    res.write_text(json.dumps({"steps": steps, "tokens": steps * 8 * 256,
                               "rows": rows}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\n=== Scaling 阶梯 (等 token 预算 {steps*8*256:,}) ===")
    print("name".ljust(6) + "stack".rjust(12) + "ppl".rjust(12) +
          "tokens/param".rjust(14) + "  diverged")
    for r in rows:
        tpp = r["stack"] and round(steps * 8 * 256 / r["stack"], 4)
        print(f"{r['name']:<6}{r['stack']:>12,}{str(r['ppl']):>12}"
              f"{str(tpp):>14}  {r['diverged']}")
    print(f"\n写出 {res}")


if __name__ == "__main__":
    main()