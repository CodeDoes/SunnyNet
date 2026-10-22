# Changelog

## v0.2.0

### 外部权重
- GLA 接入（`sunnynet/runtime/gla.py`）：门控只在 key 维，衰减可分解为按行缩放
  ⇒ 元素级递推，不需要矩阵值层级。对独立参考实现 max|diff| 2.05e-05、argmax 12/12、
  分段续算逐位 0.0。CLI `--hf-arch gla`。

### 服务与集成
- OpenAI 兼容 HTTP 服务（`sunnynet/serve.py`，标准库零依赖）：`/v1/models`、
  `/v1/chat/completions`（含 SSE 流式）、`/v1/completions`、`/health`。
  官方 `openai` SDK 四种调用方式验证通过。命令 `sunnynet-serve`。
- Ollama / LM Studio 接入（`sunnynet/integrate.py`）：`sunnynet-integrate` 生成
  Modelfile 与 LM Studio 配置。RWKV-7 GGUF 经 Ollama 真实导入并生成通过
  （`family=rwkv7`、`runner=llamacpp`）。

### C API 稳定 ABI
- `sunnynet/runtime/sn_abi.h` / `sn_abi.c`：`sn_abi_` 前缀隔离、不透明句柄、
  能力位、错误码、`SN_ABI_VERSION 1.0.0`。与内部句柄路径逐位一致（max|diff| 0.0）。
  契约测试 9 项。详见 `docs/ABI.md`。

### SIMD 后端
- AVX-512F/BW/VL 内核（`sunnynet/kernels/scan_avx512.c`）：编译通过，
  反汇编确认 198 处 `zmm` / 16 处 `vfmadd132ps`。本机 CPU 不支持，只做产物验证。
- ARM NEON 内核（`sunnynet/kernels/scan_neon.c`）：`zig cc` 交叉编译出 AArch64 ELF64，
  capstone 确认 4-lane `fmla`。执行验证放在 CI 的 `ubuntu-24.04-arm` runner。

### 权重发布
- `sunnynet-release`（`sunnynet/release.py`）：导出 safetensors（config 与训练摘要
  写入 metadata）、runtime blob、模型卡与 sha256 清单。
- 训练器支持 `--device cuda`，checkpoint 统一以 CPU tensor 落盘。
- `p100mw` 在 GTX 1060 6GB 上训练 2000 步（1.024M token），val ppl 9983 → 120.09，
  0.503 ms/token。发布资产 `.pt` 467.7 MB / `.safetensors` 421.5 MB /
  `.runtime.bin` 467.6 MB，torch、C Runtime、OpenAI 服务三条路径加载生成均通过。

### 测试
- 121 passed / 10 skipped（新增 `test_abi.py`、`test_simd.py`、`test_integrate.py`、
  `test_release.py`）。

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

### 分发
- 发布名 `sunnynet-cx`，导入名 `sunnynet`，命令 `sunnynet`。
  PyPI 上 `sunnynet` 已属他人项目（网络抓包中间件），同名不可注册。
- 发布走 GitHub Actions 的 Trusted Publisher（OIDC，无需长期令牌）；
  tag `v*` 触发 build → twine check → publish。

### 已知边界
- 内核需 AVX2/FMA 与 C 编译器（首次调用自动编译）。
- INT4 质量：英文保持，中文与代码有退化，未配校准。
- Mamba-2 / DeltaNet / GLA 目前只有 numpy 参考路径（单线程 45.7 / 91.5 / 85.1 ms/token），
  尚未 C 化；矩阵值 rank-1 递推（DeltaNet、RWKV-7）现有内核表达不了。
- 0.5B 预设在本机 6GB 显卡为边界态，未发布质量数字。