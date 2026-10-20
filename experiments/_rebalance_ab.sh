set -u
PY="${PYTHON_BIN:-python}"
R=experiments/results
mkdir -p $R

# 20M 栈参数对齐下的层深/宽度重平衡 A/B
# proto20m : dim=512 inner=896 rank=320 L=10  (栈 16,766,514)
# proto20mw: dim=640 inner=1280 rank=384 L=6  (栈 16,755,230, 偏差 +0.07%)
COMMON="--batch-size 8 --seq-len 256 --threads 8 --val-batches 8 --no-early-stop --steps 400 --lr 0.003 --warmup 60 --val-every 100"
for p in proto20m proto20mw; do
  echo "=== $p ==="
  "$PY" -X utf8 train/train.py --arch sunnynet --preset $p $COMMON \
    --out $R/rb_${p}.json 2>&1 | grep -E "params=|^  step|baseline:|Error" | tail -6
done
echo "RBDONE"