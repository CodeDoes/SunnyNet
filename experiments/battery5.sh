set -u
PY="C:/Users/陈旭/.workbuddy/binaries/python/envs/sunnynet/Scripts/python.exe"
R=experiments/results
mkdir -p $R

# 与 SunnyNet / Transformer 同协议: small 预设, T=256, batch=8, 8 线程
COMMON="--preset small --batch-size 8 --seq-len 256 --threads 8 --val-batches 8 --no-early-stop"
ARCHS="sunnynet transformer linear_attn rwkv mamba hyena h3"

echo "########## 1. 学习率筛选 (5 个 O(N) 对照, 250 步) ##########"
for a in linear_attn rwkv mamba hyena h3; do
  for lr in 0.002 0.005; do
    echo "=== $a lr=$lr ==="
    "$PY" -X utf8 train/train.py --arch $a $COMMON --steps 250 --lr $lr \
      --warmup 40 --val-every 250 --out $R/tune2_${a}_${lr}.json 2>&1 \
      | grep -E "^  step|^===|baseline:" | tail -3
  done
done

echo "########## 2. 七架构正式对比 (600 步) ##########"
for a in $ARCHS; do
  echo "=== $a ==="
  "$PY" -X utf8 train/train.py --arch $a $COMMON --steps 600 --lr 0.003 \
    --warmup 60 --val-every 150 --out $R/oncompare_${a}.json 2>&1 \
    | grep -E "^  step|^===|baseline:" | tail -6
done

echo "########## DONE ##########"