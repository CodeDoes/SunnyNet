/* SunnyNet Runtime: 分组 INT4 权重量化 + 量化 GEMV (AVX2/FMA).
 *
 * 权重布局: W 存 (out, in) 行主序, 按 in 方向每 block 个值一组共享一个 fp32 scale.
 * 量化: q = clamp(round(w / scale) + 8, 0, 15), 反量化 w' = (q - 8) * scale.
 *   scale = amax / 7 (覆盖 [0,15] 的 7 个正向档位).
 * 打包: 每 block 出 block/2 字节, 字节 k 低半字节 = 组内第 k 个值,
 *   高半字节 = 组内第 (k + block/2) 个值. 这样 maddubs 的两路输入
 *   (低/高半字节) 分别对应激活的前半段与后半段, 无需重排激活.
 *
 * 量化 GEMV 把激活也按同样 block 量化到 INT8, 用 maddubs 做点积:
 *   sum_i (q_i - 8) * x_i = sum_i q_i * x_i - 8 * sum_i x_i
 * 权重保持无符号 [0,15] 直接进 maddubs (第一操作数), 激活作有符号 (第二操作数).
 *
 * 解码是内存带宽受限 (单步读满全部权重), 4 bit/权重把读取量压到 1/8.
 */
#include "quant.h"

#include <immintrin.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

#define SN_I4_MAX_IN 8192

static inline int hsum_epi32(__m256i v) {
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    lo = _mm_add_epi32(lo, hi);
    lo = _mm_hadd_epi32(lo, lo);
    lo = _mm_hadd_epi32(lo, lo);
    return _mm_cvtsi128_si32(lo);
}

void sn_quant_i4(const float* x, unsigned char* q, float* scale,
                 int rows, int N, int block) {
    if (block <= 0) block = SN_I4_BLOCK;
    const int nb = N / block;
    const int half = block / 2;
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < rows; ++r) {
        const float* xr = x + (size_t)r * N;
        unsigned char* qr = q + (size_t)r * (N / 2);
        float* sr = scale + (size_t)r * nb;
        for (int b = 0; b < nb; ++b) {
            const int lo = b * block;
            float amax = 0.0f;
            for (int i = 0; i < block; ++i) {
                const float v = fabsf(xr[lo + i]);
                if (v > amax) amax = v;
            }
            /* q=0 对应 -8, 故只用 7 个正向档位, scale 取 amax/7 */
            const float sc = amax > 0.0f ? amax / 7.0f : 1.0f;
            const float inv = 1.0f / sc;
            sr[b] = sc;
            unsigned char* qb = qr + (size_t)b * half;
            for (int k = 0; k < half; ++k) {
                int a = (int)nearbyintf(xr[lo + k] * inv);
                int c = (int)nearbyintf(xr[lo + half + k] * inv);
                if (a < -8) a = -8; else if (a > 7) a = 7;
                if (c < -8) c = -8; else if (c > 7) c = 7;
                qb[k] = (unsigned char)((a + 8) | ((c + 8) << 4));
            }
        }
    }
}

void sn_dequant_i4(const unsigned char* q, const float* scale, float* x,
                   int rows, int N, int block) {
    if (block <= 0) block = SN_I4_BLOCK;
    const int nb = N / block;
    const int half = block / 2;
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < rows; ++r) {
        const unsigned char* qr = q + (size_t)r * (N / 2);
        const float* sr = scale + (size_t)r * nb;
        float* xr = x + (size_t)r * N;
        for (int b = 0; b < nb; ++b) {
            const float sc = sr[b];
            const unsigned char* qb = qr + (size_t)b * half;
            for (int k = 0; k < half; ++k) {
                const unsigned char byte = qb[k];
                xr[b * block + k] = (float)((int)(byte & 0x0F) - 8) * sc;
                xr[b * block + half + k] = (float)((int)(byte >> 4) - 8) * sc;
            }
        }
    }
}

void sn_gemv_i4(float* y, const float* x, const unsigned char* Wq,
                const float* scale, int in, int out, int block) {
    if (block <= 0) block = SN_I4_BLOCK;
    if (in % block || in > SN_I4_MAX_IN) { sn_gemv_f32(y, x, (const float*)Wq, in, out); return; }
    const int nb = in / block;
    const int half = block / 2;
    const int vchunks = half / 32;      /* 每 block 的 AVX2 字节块数 */
    const int vtail = half % 32;        /* 尾部字节数 */

    signed char xq[SN_I4_MAX_IN];
    float xs[128];
    int xsum[128];
    for (int b = 0; b < nb; ++b) {
        const int lo = b * block;
        float amax = 0.0f;
        for (int i = 0; i < block; ++i) {
            const float v = fabsf(x[lo + i]);
            if (v > amax) amax = v;
        }
        const float sc = amax > 0.0f ? amax / 127.0f : 1.0f;
        const float inv = 1.0f / sc;
        xs[b] = sc;
        int sum = 0;
        for (int i = 0; i < block; ++i) {
            int t = (int)nearbyintf(x[lo + i] * inv);
            if (t > 127) t = 127; else if (t < -127) t = -127;
            xq[lo + i] = (signed char)t;
            sum += t;
        }
        xsum[b] = sum;
    }

    const __m256i ones = _mm256_set1_epi16(1);
    #pragma omp parallel for schedule(static) if(out >= 512)
    for (int o = 0; o < out; ++o) {
        const unsigned char* row = Wq + (size_t)o * (in / 2);
        const float* sr = scale + (size_t)o * nb;
        const __m256i mask = _mm256_set1_epi8(0x0F);
        float acc = 0.0f;
        for (int b = 0; b < nb; ++b) {
            const unsigned char* wb = row + (size_t)b * half;
            const signed char* xlo = xq + b * block;
            const signed char* xhi = xq + b * block + half;
            __m256i p16 = _mm256_setzero_si256();
            for (int c = 0; c < vchunks; ++c) {
                const __m256i w = _mm256_loadu_si256((const __m256i*)(wb + 32 * c));
                const __m256i wlo = _mm256_and_si256(w, mask);
                const __m256i whi = _mm256_and_si256(_mm256_srli_epi16(w, 4), mask);
                const __m256i alo = _mm256_loadu_si256((const __m256i*)(xlo + 32 * c));
                const __m256i ahi = _mm256_loadu_si256((const __m256i*)(xhi + 32 * c));
                p16 = _mm256_add_epi16(p16, _mm256_maddubs_epi16(wlo, alo));
                p16 = _mm256_add_epi16(p16, _mm256_maddubs_epi16(whi, ahi));
            }
            const __m256i p32 = _mm256_madd_epi16(p16, ones);
            int s = hsum_epi32(p32);
            for (int k = 0; k < vtail; ++k) {
                const int j = vchunks * 32 + k;
                const unsigned char byte = wb[j];
                s += (int)(byte & 0x0F) * (int)xlo[j];
                s += (int)(byte >> 4) * (int)xhi[j];
            }
            acc += sr[b] * xs[b] * (float)(s - 8 * xsum[b]);
        }
        y[o] = acc;
    }
}

void sn_gemv_f32(float* y, const float* x, const float* W, int in, int out) {
    #pragma omp parallel for schedule(static) if(out >= 512)
    for (int o = 0; o < out; ++o) {
        const float* row = W + (size_t)o * in;
        __m256 acc = _mm256_setzero_ps();
        int i = 0;
        for (; i + 8 <= in; i += 8)
            acc = _mm256_fmadd_ps(_mm256_loadu_ps(row + i), _mm256_loadu_ps(x + i), acc);
        __m128 lo = _mm256_castps256_ps128(acc);
        __m128 hi = _mm256_extractf128_ps(acc, 1);
        lo = _mm_add_ps(lo, hi);
        lo = _mm_hadd_ps(lo, lo);
        lo = _mm_hadd_ps(lo, lo);
        float s = _mm_cvtss_f32(lo);
        for (; i < in; ++i) s += row[i] * x[i];
        y[o] = s;
    }
}