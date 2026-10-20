set -u
PY="${PYTHON_BIN:-python}"
R=experiments/results
mkdir -p $R

PRESET="${1:-proto20m}"
STEPS="${2:-600}"
T="${3:-256}"
TAG="${4:-20m}"
COMMON="--preset $PRESET --batch-size 8 --seq-len $T --threads 8 --val-batches 8 --no-early-stop --steps $STEPS --lr 0.003 --warmup 60 --val-every 150"

echo "########## 同协议对比 preset=$PRESET steps=$STEPS T=$T ##########"
for a in sunnynet mamba h3 hyena rwkv linear_attn transformer; do
  echo "=== $a ==="
  "$PY" -X utf8 train/train.py --arch $a $COMMON --out $R/sc_${TAG}_${a}.json 2>&1 \
    | grep -E "^  step|^===|baseline:|Error|error" | tail -6
done

echo "########## DONE ##########"