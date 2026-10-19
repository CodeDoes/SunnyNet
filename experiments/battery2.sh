set -u
PY="C:/Users/陈旭/.workbuddy/binaries/python/envs/sunnynet/Scripts/python.exe"
R=experiments/results
mkdir -p $R
run(){ echo "=== $* "; "$PY" -X utf8 train/train.py "$@" 2>&1 | grep -E "^  step|^===|访存|params|早停"; }

echo "########## 20M 缩放对照 (300 步, T=512) ##########"
run --arch sunnynet    --preset proto20m --steps 300 --batch-size 4 --seq-len 512 --lr 0.002 --warmup 40 --threads 8 --val-every 100 --val-batches 6 --out $R/scale_proto20m_sunnynet.json
run --arch transformer --preset proto20m --steps 300 --batch-size 4 --seq-len 512 --lr 0.004 --warmup 40 --threads 8 --val-every 100 --val-batches 6 --out $R/scale_proto20m_transformer.json

echo "########## 长序列训练对照 (T=1024, 200 步) ##########"
run --arch sunnynet    --preset small --steps 200 --batch-size 4 --seq-len 1024 --lr 0.003 --warmup 30 --threads 8 --val-every 100 --val-batches 4 --out $R/longseq_sunnynet_T1024.json
run --arch transformer --preset small --steps 200 --batch-size 4 --seq-len 1024 --lr 0.006 --warmup 30 --threads 8 --val-every 100 --val-batches 4 --out $R/longseq_transformer_T1024.json

echo "########## DONE ##########"