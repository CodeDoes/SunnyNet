# CPU 推理镜像: 内核在 builder 阶段编好再拷入运行层, 运行层不带 gcc 与头文件。
FROM python:3.11-slim AS builder

ARG SUNNYNET_VERSION=0.3.0

# Linux 侧 _prebuilt 只有 win-x86_64, 内核必须本机编。pip 走 sdist 时 setuptools
# 的 build_py 调 _native.build_prebuilt, 用 gcc 产出 linux-x86_64 的 .so。
RUN apt-get update \
 && apt-get install -y --no-install-recommends gcc libc6-dev \
 && rm -rf /var/lib/apt/lists/*

# torch 用官方 CPU index (同一条命令解析其纯 Python 依赖); sunnynet-cx 的依赖
# 已备齐, 故 --no-deps, 避免再从 PyPI 拉 CUDA 版 torch。
RUN pip install --no-cache-dir --prefix=/install \
      --index-url https://download.pytorch.org/whl/cpu torch
RUN pip install --no-cache-dir --prefix=/install numpy
RUN pip install --no-cache-dir --prefix=/install --no-deps \
      "sunnynet-cx==${SUNNYNET_VERSION}"

# 缺库即构建失败, 不留给运行期静默回退源码编译 (运行层无编译器)。
RUN python -c "import importlib,sys;from sunnynet import _native as n; \
miss=[d for d in (importlib.import_module(x).DLL.name for x in n.MODULES) if n.find(d) is None]; \
print('prebuilt missing:', miss); sys.exit(1 if miss else 0)"

FROM python:3.11-slim

# .so 以 -fopenmp 编出, 动态链接 libgomp
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

ENV PYTHONUNBUFFERED=1

EXPOSE 8080

# 权重与分词器不进镜像 (体积与许可), 运行期挂载后由 --ckpt / --tokenizer 指认:
#   docker run --rm -p 8080:8080 -v $PWD/models:/models \
#     ghcr.io/sunnyclaw-cx/sunnynet \
#     --ckpt /models/proto20m.pt --tokenizer /models/tokenizer.json --host 0.0.0.0
ENTRYPOINT ["sunnynet-serve"]
CMD ["--host", "0.0.0.0", "--port", "8080"]