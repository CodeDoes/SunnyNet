set -u
PY="${PYTHON_BIN:-python}"
R=experiments/results
mkdir -p $R

echo "########## 训练收敛 checkpoint (small, 3000 步) ##########"
"$PY" -X utf8 train/train.py --arch sunnynet --preset small --steps 3000 \
  --batch-size 8 --seq-len 256 --lr 0.003 --warmup 120 --threads 8 \
  --val-every 500 --val-batches 8 --save-ckpt runs/small_conv.pt \
  --out $R/conv_sunnynet.json 2>&1 | grep -E "^  step|^===|早停"

echo "########## 长距离探针 + 域困惑度 ##########"
"$PY" -X utf8 train/evaluate.py --ckpt runs/small_conv.pt \
  --seq-len 512 --windows 24 --recall-seqs 6 --gen \
  --out $R/eval_sunnynet.json 2>&1 | tail -30

echo "########## DONE ##########"