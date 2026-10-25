/* 在 x86 上跑真 AVX-512 源码用的标量 shim.
 * 本机 CPU 无 AVX-512F, 真指令无法执行; 此处的 __m512/__m512i 是 16 元数组,
 * 逐元素运算与 AVX-512 语义一致。置换用真实索引向量, 所以组内 lane 语义
 * (低 8 = 组 A, 高 8 = 组 B) 走的是源码里的真索引, 不是重写。 */
#ifndef SN_AVX512_SHIM_H
#define SN_AVX512_SHIM_H

typedef float __m512 __attribute__((vector_size(64)));
typedef int __m512i __attribute__((vector_size(64)));
typedef unsigned short __mmask16;

static inline __m512 _mm512_set1_ps(float x) {
    return (__m512){x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x};
}
static inline __m512 _mm512_setzero_ps(void) {
    return (__m512){0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
}
static inline __m512 _mm512_loadu_ps(const float* p) {
    __m512 v; for (int i = 0; i < 16; ++i) v[i] = p[i]; return v;
}
static inline void _mm512_storeu_ps(float* p, __m512 v) {
    for (int i = 0; i < 16; ++i) p[i] = v[i];
}
static inline __m512 _mm512_add_ps(__m512 a, __m512 b) { return a + b; }
static inline __m512 _mm512_sub_ps(__m512 a, __m512 b) { return a - b; }
static inline __m512 _mm512_mul_ps(__m512 a, __m512 b) { return a * b; }
static inline __m512 _mm512_div_ps(__m512 a, __m512 b) { return a / b; }
static inline __m512 _mm512_fmadd_ps(__m512 a, __m512 b, __m512 c) {
    return a * b + c;
}

static inline __m512i _mm512_loadu_si512(const void* p) {
    __m512i v; const int* q = (const int*)p;
    for (int i = 0; i < 16; ++i) v[i] = q[i];
    return v;
}

static inline __m512 _mm512_permutexvar_ps(__m512i idx, __m512 a) {
    __m512 r;
    for (int i = 0; i < 16; ++i) r[i] = a[idx[i] & 15];
    return r;
}

static inline __m512 _mm512_maskz_loadu_ps(__mmask16 m, const float* p) {
    __m512 v = _mm512_setzero_ps();
    for (int i = 0; i < 16; ++i) if (m & (1u << i)) v[i] = p[i];
    return v;
}
static inline void _mm512_mask_storeu_ps(float* p, __mmask16 m, __m512 v) {
    for (int i = 0; i < 16; ++i) if (m & (1u << i)) p[i] = v[i];
}

/* 尾部函数走 AVX2 (8 lane), 一并给 shim, 否则源码编不过。 */
typedef float __m256 __attribute__((vector_size(32)));
typedef int __m256i __attribute__((vector_size(32)));

static inline __m256 _mm256_set1_ps(float x) {
    return (__m256){x, x, x, x, x, x, x, x};
}
static inline __m256 _mm256_setzero_ps(void) {
    return (__m256){0, 0, 0, 0, 0, 0, 0, 0};
}
static inline __m256i _mm256_setr_epi32(int a, int b, int c, int d,
                                        int e, int f, int g, int h) {
    return (__m256i){a, b, c, d, e, f, g, h};
}
static inline __m256 _mm256_loadu_ps(const float* p) {
    __m256 v; for (int i = 0; i < 8; ++i) v[i] = p[i]; return v;
}
static inline void _mm256_storeu_ps(float* p, __m256 v) {
    for (int i = 0; i < 8; ++i) p[i] = v[i];
}
static inline __m256 _mm256_add_ps(__m256 a, __m256 b) { return a + b; }
static inline __m256 _mm256_sub_ps(__m256 a, __m256 b) { return a - b; }
static inline __m256 _mm256_mul_ps(__m256 a, __m256 b) { return a * b; }
static inline __m256 _mm256_div_ps(__m256 a, __m256 b) { return a / b; }
static inline __m256 _mm256_fmadd_ps(__m256 a, __m256 b, __m256 c) {
    return a * b + c;
}
static inline __m256 _mm256_permutevar8x32_ps(__m256 a, __m256i idx) {
    __m256 r;
    for (int i = 0; i < 8; ++i) r[i] = a[idx[i] & 7];
    return r;
}
#endif