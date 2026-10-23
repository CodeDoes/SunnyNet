"""维度甜点扫描: 同栈参数量下, 主干维度 d 对训练步耗时与质量的影响.

§2 实测 CPU GEMM 甜点区在 n<=128; §3 实测工作集需 <=3-4MB。
本脚本扫描 d, 看训练步 ms/token 是否存在 CPU 侧极小值——若有, 即为
"CPU 原生维度"的直接证据; 若无, 则说明维度不是主要杠杆 (须如实记录)。

口径: 每格独立子进程 (见测量纪律), 同 T/B/步数/学习率, 取最终 val ppl 与 ms/token。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
CONFIGS = [
    ("d64", 64, 128, 192, 4),
    ("d96", 96, 192, 152, 4),
    ("d128", 128, 256, 192, 4),
    ("d192", 192, 384, 256, 4),
]


def run(name, dim, inner, rank, layers, steps, seq, bs, lr):
    out = Path(f"/tmp/_dim_{name}.json")
    cfg = (f"vocab_size=8192,dim={dim},inner={inner},rank={rank},"
           f"layers={layers},h_max=8192.0")
    code = f"""
import json,sys,math,time
sys.path.insert(0,{str(ROOT)!r}); sys.path.insert(0,{str(ROOT/'train')!r})
import torch
from sunnynet.model import SunnyConfig, SunnyNet
from train import DomainMixer, evaluate, lr_at
torch.manual_seed(1234); torch.set_num_threads(6)
cfg = SunnyConfig(**dict(x.split('=') for x in {cfg!r}.split(',')))
cfg.vocab_size=int(cfg.vocab_size); cfg.dim=int(cfg.dim); cfg.inner=int(cfg.inner)
cfg.rank=int(cfg.rank); cfg.layers=int(cfg.layers); cfg.h_max=float(cfg.h_max)
cfg.scan_chunk=64
m = SunnyNet(cfg)
mix = DomainMixer({{'zh_ancient':0.40,'en_literature':0.35,'code_python':0.25}},{seq},{bs},'train',seed=1234)
vmix = DomainMixer({{'zh_ancient':0.40,'en_literature':0.35,'code_python':0.25}},{seq},{bs},'val',seed=1)
vb = vmix.val_batches(4)
opt = torch.optim.AdamW(m.parameters(), lr={lr}, betas=(0.9,0.95), weight_decay=0.01)
t0=time.perf_counter(); ntok=0
for step in range({steps}):
    for g in opt.param_groups: g['lr']=lr_at(step,{steps},{lr},20)
    x,y = mix.batch()
    logits = m(x)
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1,8192), y.reshape(-1))
    opt.zero_grad(set_to_none=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
    ntok += x.numel()
dt=time.perf_counter()-t0
vl = evaluate(m, vb)
json.dump({{'preset':'{name}','dim':{dim},'layers':{layers},'params':m.num_params(),
  'ms_per_token':dt/ntok*1e3,'val_loss':vl,'ppl':math.exp(min(vl,20))}},
  open(r'{out}','w'))
print('OK','{name}')
"""
    r = subprocess.run([PY, "-X", "utf8", "-c", code], capture_output=True)
    if r.returncode != 0:
        print(f"  {name} 失败: {r.stderr.decode('utf-8','replace')[-300:]}")
        return None
    return json.loads(out.read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-3)
    args = ap.parse_args()

    print(f"维度扫描  steps={args.steps} T={args.seq_len} B={args.batch_size} lr={args.lr}")
    print(f"  {'name':>6} {'dim':>5} {'L':>3} {'params':>12} {'ms/tok':>10} {'ppl':>9}")
    rows = []
    for name, d, inner, rank, L in CONFIGS:
        r = run(name, d, inner, rank, L, args.steps, args.seq_len,
                args.batch_size, args.lr)
        if r:
            rows.append(r)
            print(f"  {name:>6} {r['dim']:>5} {r['layers']:>3} {r['params']:>12,} "
                  f"{r['ms_per_token']:>10.4f} {r['ppl']:>9.2f}")
    if rows:
        best_s = min(rows, key=lambda r: r["ms_per_token"])
        best_q = min(rows, key=lambda r: r["ppl"])
        print(f"\n最快: {best_s['preset']} ({best_s['ms_per_token']:.4f} ms/tok)")
        print(f"最优: {best_q['preset']} (ppl {best_q['ppl']:.2f})")


if __name__ == "__main__":
    main()