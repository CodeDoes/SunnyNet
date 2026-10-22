# Changelog

## v0.1.0

首个可用版本。

### 原创架构
- SunnyNet：CPU 上训练的通用序列架构（非 Transformer / RNN / Mamba / CNN）。
  CenteredScaleNorm、组内成对通道竞争、对数域并行扫描、多尺度衰减。
- 预设 `proto20mw`（d640/i1280/r384/L6）与 `p100mw`（d1408/i1920/r1024/L8）。
- 不变量自检（`tests/test_architecture.py`）：fp64 单步 vs 整段、分段续算、分块不变性、因果性。

### 手写内核
- AVX2/FMA 融合内核（`sunnynet/kernels/`），训练步 1.10–1.17×。
- 通用一阶线性递推 C Runtime（`sunnynet/runtime/recur.c`），对 torch 向量化基线中位 9.90×。

### 推理 Runtime
- 整模型 C 单步解码（`sunnynet/runtime/sunnynet.c`）：B=1 单线程 T=256/1024/2048 为 1.53×/2.05×/1.97×。
- GGUF 只读（`sunnynet/runtime/gguf.py`），外部权重反量化与官方 `.pth` 逐张量一致。
- RWKV-7 整模型 C 内核（`sunnynet/runtime/rwkv7.c`）：对 numpy 参考 5.39×，8 线程 7.49×；
  单步 logits max|diff| 7.25e-05，120 步 argmax 全一致。
- INT4 分组权重量化（`sunnynet/runtime/quant.c`）：端到端 2.83×，体积 729→269 MB。
- 外部权重接入：Mamba-2（`runtime/mamba2.py`）与 DeltaNet（`runtime/deltanet.py`）
  逐 token 单步递推，不经 PyTorch 前向。Mamba-2 对 `transformers` 官方实现
  argmax 24/24；DeltaNet 贪心续写连贯（"The capital of France is the city of Paris…"）。

### 命令行
- `sunnynet` / `python -m sunnynet.cli`：`--ckpt`（SunnyNet）、`--gguf`（外部权重）、
  `--hf-dir`（Mamba-2 / DeltaNet）、`--backend c`、`--gguf-backend c`、`--quant {fp32,int4}`。
- `sunnynet.runtime.load_model(arch, path)`：按架构名加载外部权重。

### 已知边界
- 内核需 AVX2/FMA 与 C 编译器（首次调用自动编译）。
- INT4 质量：英文保持，中文与代码有退化，未配校准。
- Mamba-2 / DeltaNet 目前只有 numpy 参考路径（单线程 45.7 / 91.5 ms/token），
  尚未 C 化；矩阵值 rank-1 递推（DeltaNet、RWKV-7）现有内核表达不了。
- 0.5B 预设在本机 6GB 显卡为边界态，未发布质量数字。