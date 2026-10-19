set -u
PY="C:/Users/陈旭/.workbuddy/binaries/python/envs/sunnynet/Scripts/python.exe"
R=experiments/results
mkdir -p $R

echo "########## 1. 探针 checkpoint: SunnyNet (T=1024) ##########"
"$PY" -X utf8 train/train.py --arch sunnynet --preset small --steps 800 \
  --batch-size 4 --seq-len 1024 --lr 0.003 --warmup 100 --threads 8 \
  --val-every 100 --val-batches 6 --no-early-stop \
  --out $R/probe_train_sunnynet.json --save-ckpt data/ck_probe_sunnynet.pt

echo "########## 2. 探针 checkpoint: Transformer (T=1024) ##########"
"$PY" -X utf8 train/train.py --arch transformer --preset small --steps 800 \
  --batch-size 4 --seq-len 1024 --lr 0.004 --warmup 100 --threads 8 \
  --val-every 100 --val-batches 6 --no-early-stop \
  --out $R/probe_train_transformer.json --save-ckpt data/ck_probe_transformer.pt

echo "########## 3. 长距离依赖探针 (seq_len=1024) ##########"
"$PY" -X utf8 train/evaluate.py --ckpt data/ck_probe_sunnynet.pt \
  --seq-len 1024 --windows 16 --recall-seqs 8 --skip-latency \
  --out $R/longrange_sunnynet.json

"$PY" -X utf8 train/evaluate.py --ckpt data/ck_probe_transformer.pt \
  --seq-len 1024 --windows 16 --recall-seqs 8 --skip-latency \
  --out $R/longrange_transformer.json

echo "########## DONE ##########"