#!/usr/bin/env bash
# 交叉编译 NEON 内核给 ARM 端侧设备 (树莓派 5 / RK3588 / 安卓).
#
# 用法:
#   scripts/build_arm.sh                       # 用 zig 编 aarch64-linux-gnu
#   scripts/build_arm.sh aarch64-linux-musl    # 静态链接, 适合更精简的 rootfs
#   scripts/build_arm.sh arm-linux-gnueabihf   # 32 位 ARMv7 (树莓派 3/4 的 32 位系统)
#
# 产出: sunnynet/kernels/_build/scan_neon.<target>.so
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TARGET="${1:-aarch64-linux-gnu}"
SRC="$ROOT/sunnynet/kernels/scan_neon.c"
OUT_DIR="$ROOT/sunnynet/kernels/_build"
OUT="$OUT_DIR/sunnynet_neon.$TARGET.so"

mkdir -p "$OUT_DIR"

# 交互环境里 PATH 上常常没有 python; 依次试 PYTHON、python3、python、py。
find_py() {
  local c
  for c in "${PYTHON:-}" python3 python py; do
    [ -n "$c" ] || continue
    if command -v "$c" >/dev/null 2>&1; then echo "$c"; return; fi
  done
}

find_zig() {
  if command -v zig >/dev/null 2>&1; then echo zig; return; fi
  local py="$1" z=""
  # pip 安装的 ziglang 包
  if [ -n "$py" ]; then
    z="$("$py" - <<'PY'
import pathlib, sys
try:
    import ziglang
    p = pathlib.Path(ziglang.__file__).parent / ("zig.exe" if sys.platform == "win32" else "zig")
    if p.exists():
        print(p)
except Exception:
    pass
PY
)"
  fi
  # 没有 python 时的兜底: 扫常见 site-packages 位置
  if [ -z "$z" ]; then
    z="$(find "${HOME:-/root}" /opt -maxdepth 6 -type f \( -name zig -o -name zig.exe \) 2>/dev/null)"
    z="${z%%$'\n'*}"
  fi
  echo "$z"
}

PY_BIN="$(find_py)"
ZIG="$(find_zig "$PY_BIN")"
if [ -z "$ZIG" ]; then
  echo "未找到 zig。安装: pip install ziglang  (或 apt install zig)" >&2
  exit 1
fi

echo "[build_arm] target=$TARGET"
echo "[build_arm] zig=$ZIG"

# 32 位 ARMv7 需要显式开 NEON 与 VFPv4; AArch64 的 NEON 是基线特性, 加 -mfpu 反而报错。
EXTRA=""
case "$TARGET" in
  arm-*|thumb-*) EXTRA="-mfpu=neon-vfpv4 -mfloat-abi=hard" ;;
esac

"$ZIG" cc -target "$TARGET" -O3 -shared -fPIC -std=c11 \
  -DNDEBUG $EXTRA "$SRC" -o "$OUT"

echo "[build_arm] 产出 $OUT ($(wc -c <"$OUT") bytes)"
echo "[build_arm] 校验架构:"
"$PY_BIN" - "$OUT" <<'PY'
import struct, sys
d = open(sys.argv[1], 'rb').read()
m = struct.unpack_from('<H', d, 18)[0]
names = {0x28: "ARM(32)", 0xB7: "AArch64"}
print(f"  ELF machine = {hex(m)} {names.get(m, '?')}")
for sym in (b"step_core_neon_f32", b"core_neon_fwd_f32", b"scan_neon_f32",
            b"sn_neon_isa"):
    print(f"  {sym.decode():22s} {'ok' if sym in d else 'MISSING'}")
PY