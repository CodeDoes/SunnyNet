/* SunnyNet 稳定 C ABI 实现. 见 sn_abi.h 的契约说明.
 *
 * 本文件把内部内核 (recur.c / rwkv7.c) 包一层带 sn_abi_ 前缀、带参数校验与
 * 错误码的稳定接口。内部符号不属于稳定接口, 可自由改动。
 */
#include "sn_abi.h"

#include <stdlib.h>
#include <string.h>

/* 内部接口 (非稳定). */
#include "recur.h"
#include "rwkv7.h"

#define SN_ABI_VERSION_STR "1.0.0"

int32_t sn_abi_version(void) { return SN_ABI_VERSION; }

uint32_t sn_abi_caps(void) {
    return SN_CAP_RECUR_SCAN | SN_CAP_RECUR_STEP | SN_CAP_QUANT_I8 |
           SN_CAP_RWKV7 | SN_CAP_THREADS;
}

sn_status sn_abi_check(int32_t caller_major) {
    return caller_major == SN_ABI_VERSION_MAJOR ? SN_OK : SN_E_UNSUPPORTED;
}

static int32_t copy_str(const char* src, char* buf, int32_t buflen) {
    int32_t need = (int32_t)strlen(src) + 1;
    if (buf == NULL || buflen < need) return need;
    memcpy(buf, src, (size_t)need);
    return need;
}

int32_t sn_abi_version_string(char* buf, int32_t buflen) {
    return copy_str(SN_ABI_VERSION_STR, buf, buflen);
}

int32_t sn_abi_backend_string(char* buf, int32_t buflen) {
    return copy_str("c", buf, buflen);
}

sn_status sn_abi_set_threads(int32_t n) {
    if (n <= 0) return SN_E_INVAL;
    sn_set_threads((int)n);
    return SN_OK;
}

int32_t sn_abi_get_threads(void) { return (int32_t)sn_threads(); }

/* ------------------------------------------------------------ 递推内核 */
static int dims_bad(const void* p, int32_t a, int32_t b) {
    return p == NULL || a <= 0 || b <= 0;
}

sn_status sn_abi_recur_step(const float* a, const float* b, float* s,
                            int32_t B, int32_t N, int32_t gated) {
    if (dims_bad(a, B, N) || b == NULL || s == NULL) return SN_E_INVAL;
    sn_recur_step(a, b, s, (int)B, (int)N, gated ? 1 : 0);
    return SN_OK;
}

sn_status sn_abi_recur_scan(const float* a, const float* b, const float* s0,
                            float* out, float* send,
                            int32_t B, int32_t T, int32_t N, int32_t gated) {
    if (dims_bad(a, B, N) || b == NULL || s0 == NULL || out == NULL ||
        send == NULL || T <= 0)
        return SN_E_INVAL;
    sn_recur_scan(a, b, s0, out, send, (int)B, (int)T, (int)N, gated ? 1 : 0);
    return SN_OK;
}

/* -------------------------------------------------------------- 量化 */
sn_status sn_abi_quant_i8(const float* x, int8_t* q, float* scale,
                          int32_t B, int32_t N, int32_t block) {
    if (dims_bad(x, B, N) || q == NULL || scale == NULL) return SN_E_INVAL;
    sn_quant_i8(x, q, scale, (int)B, (int)N, (int)block);
    return SN_OK;
}

sn_status sn_abi_dequant_i8(const int8_t* q, const float* scale, float* x,
                            int32_t B, int32_t N, int32_t block) {
    if (dims_bad(x, B, N) || q == NULL || scale == NULL) return SN_E_INVAL;
    sn_dequant_i8(q, scale, x, (int)B, (int)N, (int)block);
    return SN_OK;
}

/* ---------------------------------------------------------- RWKV-7 */
/* 独立句柄名, 与被包装的内部类型区分; 布局不对外暴露. */
struct SnAbiRwkv7 {
    SnRWKV7* inner;
};

SnAbiRwkv7* sn_abi_rwkv7_load(const char* blob_path) {
    if (blob_path == NULL) return NULL;
    SnRWKV7* inner = sn_rk_load(blob_path);
    if (inner == NULL) return NULL;
    SnAbiRwkv7* m = (SnAbiRwkv7*)malloc(sizeof(SnAbiRwkv7));
    if (m == NULL) {
        sn_rk_free(inner);
        return NULL;
    }
    m->inner = inner;
    return m;
}

int32_t sn_abi_rwkv7_vocab(const SnAbiRwkv7* m) {
    return (m == NULL) ? 0 : (int32_t)sn_rk_vocab(m->inner);
}

sn_status sn_abi_rwkv7_step(SnAbiRwkv7* m, int32_t token, float* logits) {
    if (m == NULL || logits == NULL || token < 0) return SN_E_INVAL;
    sn_rk_step(m->inner, (int)token, logits);
    return SN_OK;
}

sn_status sn_abi_rwkv7_prefill(SnAbiRwkv7* m, const int32_t* tokens,
                               int32_t n, float* logits) {
    if (m == NULL || tokens == NULL || logits == NULL || n <= 0)
        return SN_E_INVAL;
    /* int32_t 与 int 在本平台同宽, 直接传入; 若不同宽需逐元素转换. */
    sn_rk_prefill(m->inner, tokens, (int)n, logits);
    return SN_OK;
}

sn_status sn_abi_rwkv7_reset(SnAbiRwkv7* m) {
    if (m == NULL) return SN_E_INVAL;
    sn_rk_reset(m->inner);
    return SN_OK;
}

void sn_abi_rwkv7_free(SnAbiRwkv7* m) {
    if (m == NULL) return;
    sn_rk_free(m->inner);
    free(m);
}