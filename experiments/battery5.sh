set -u
PY="${PYTHON_BIN:-python}"
R=experiments/results
mkdir -p $R

# 同协议: small 预设, T=256, batch=8, 8 线程, 600 步, lr=0.003
COMMON="--preset small --batch-size 8 --seq-len 256 --threads 8 --val-batches 8 --no-early-stop --steps 600 --lr 0.003 --warmup 60 --val-every 150"

echo "########## 七架构同协议对比 (600 步) ##########"
for a in sunnynet transformer linear_attn rwkv mamba hyena h3; do
  echo "=== $a ==="
  "$PY" -X utf8 train/train.py --arch $a $COMMON --out $R/oncompare_${a}.json 2>&1 \
    | grep -E "^  step|^===|baseline:" | tail -6
done

echo "########## DONE ##########"