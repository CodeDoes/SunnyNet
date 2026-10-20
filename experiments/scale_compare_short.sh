set -u
# 20M 规模五架构同协议对照 (短协议, 用栈参数对齐的重平衡配置驱动)
# 注: 同行 d_model 对齐用 train.py 内置的 build_linear_baseline (目标=SunnyNet 栈参数)
PY="${PYTHON_BIN:-python}"
R=experiments/results
mkdir -p $R

PRESET="${1:-proto20mw}"
STEPS="${2:-200}"
T="${3:-256}"
TAG="${4:-20mw}"

COMMON="--preset $PRESET --batch-size 8 --seq-len $T --threads 8 --val-batches 6 --no-early-stop --steps $STEPS --lr 0.003 --warmup 40 --val-every 50"
for a in sunnynet mamba h3 hyena rwkv; do
  echo "=== $a ==="
  "$PY" -X utf8 train/train.py --arch $a $COMMON --out $R/sc_${TAG}_${a}.json 2>&1 \
    | grep -aE "params=|baseline:|^  step|Error|Trace" | tail -4
done
echo "SCDONE"