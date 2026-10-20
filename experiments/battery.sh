set -u
PY="${PYTHON_BIN:-python}"
R=experiments/results
mkdir -p $R
run(){ echo "=== $* "; "$PY" -X utf8 train/train.py "$@" 2>&1 | grep -E "^  step|^===|基线|访存|params|早停"; }

echo "########## 1. 基线对照 (各自最优 lr, 独占运行) ##########"
run --arch sunnynet    --preset small --steps 1200 --batch-size 8 --seq-len 256 --lr 0.003 --warmup 80 --threads 8 --val-every 150 --val-batches 8 --out $R/baseline_sunnynet.json
run --arch transformer --preset small --steps 1200 --batch-size 8 --seq-len 256 --lr 0.006 --warmup 80 --threads 8 --val-every 150 --val-batches 8 --out $R/baseline_transformer.json

echo "########## 2. 消融 (600 步) ##########"
run --arch sunnynet --preset small --steps 600 --batch-size 8 --seq-len 256 --lr 0.003 --warmup 60 --threads 8 --val-every 150 --val-batches 8 --out $R/ablation_full.json
for v in "decay_gate=0:no_decay_gate" "circulant=0:no_circulant" "local_taps=0:no_local_taps" "current=0:no_current" "decay_gate=0,circulant=0,local_taps=0,current=0:none"; do
  ab="${v%%:*}"; nm="${v##*:}"
  run --arch sunnynet --preset small --steps 600 --batch-size 8 --seq-len 256 --lr 0.003 --warmup 60 --threads 8 --val-every 150 --val-batches 8 --ablation "$ab" --out $R/ablation_$nm.json
done

echo "########## 3. 种子稳定性 ##########"
for sd in 1234 2024 3407; do
  run --arch sunnynet --preset small --steps 600 --batch-size 8 --seq-len 256 --lr 0.003 --warmup 60 --threads 8 --val-every 150 --val-batches 8 --seed $sd --out $R/seed_$sd.json
done

echo "########## 4. 缩放: 20M 原型 ##########"
run --arch sunnynet    --preset proto20m --steps 800 --batch-size 4 --seq-len 512 --lr 0.002 --warmup 80 --threads 8 --val-every 200 --val-batches 6 --out $R/scale_proto20m_sunnynet.json
run --arch transformer --preset proto20m --steps 800 --batch-size 4 --seq-len 512 --lr 0.004 --warmup 80 --threads 8 --val-every 200 --val-batches 6 --out $R/scale_proto20m_transformer.json

echo "########## DONE ##########"
