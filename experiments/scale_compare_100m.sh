set -u
# 100M 规模五架构同协议对照
# SunnyNet: p100mw (dim=1408 inner=1920 rank=1024 L=8, 栈 93,828,904)
# 同行 d_model 由 train.py 的 build_linear_baseline 按栈参数对齐 (n_layers=8)
PY="${PYTHON_BIN:-python}"
R=experiments/results
mkdir -p $R

PRESET="${1:-p100mw}"
STEPS="${2:-150}"
T="${3:-256}"
TAG="${4:-100m}"

COMMON="--preset $PRESET --batch-size 8 --seq-len $T --threads 8 --val-batches 6 --no-early-stop --steps $STEPS --lr 0.003 --warmup 30 --val-every 50"
for a in sunnynet mamba h3 hyena rwkv; do
  echo "=== $a ==="
  "$PY" -X utf8 train/train.py --arch $a $COMMON --out $R/sc_${TAG}_${a}.json 2>&1 \
    | grep -aE "params=|baseline:|^  step|Error|Trace" | tail -4
done
echo "SC100DONE"