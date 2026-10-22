/* fp16 权重内核的加载宽度对比: 128 位 vs 256 位加载。
 *
 * 128 位版每次读 8 个 half (16B) 转 8 个 float 做 FMA。
 * 256 位版一次读 16 个 half (32B), 拆成两个 __m128i 各转 8 个 float,
 * 恢复与 fp32 相同的 32B 加载宽度。
 *
 * 编译: gcc -O3 -mavx2 -mfma -march=x86-64-v3 -funroll-loops -fopenmp
 *       bench_f16b.c -o bench_f16b.exe
 */
#include <immintrin.h>
#include <math.h>
#include <malloc.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void gemv_f32(float* y, const float* x, const float* W, int in, int out) {
#pragma omp parallel for schedule(static)
    for (int j = 0; j < out; ++j) {
        const float* row = W + (size_t)j * in;
        __m256 acc = _mm256_setzero_ps();
        int i = 0;
        for (; i + 8 <= in; i += 8)
            acc = _mm256_fmadd_ps(_mm256_loadu_ps(row + i),
                                  _mm256_loadu_ps(x + i), acc);
        float t[8];
        _mm256_storeu_ps(t, acc);
        float s = t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
        for (; i < in; ++i) s += row[i] * x[i];
        y[j] = s;
    }
}

/* 128 位加载: 8 half/次 */
static void gemv_f16_128(float* y, const float* x, const unsigned short* W,
                         int in, int out) {
#pragma omp parallel for schedule(static)
    for (int j = 0; j < out; ++j) {
        const unsigned short* row = W + (size_t)j * in;
        __m256 acc = _mm256_setzero_ps();
        int i = 0;
        for (; i + 8 <= in; i += 8)
            acc = _mm256_fmadd_ps(
                _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(row + i))),
                _mm256_loadu_ps(x + i), acc);
        float t[8];
        _mm256_storeu_ps(t, acc);
        float s = t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
        for (; i < in; ++i) s += (float)row[i] * x[i];
        y[j] = s;
    }
}

/* 256 位加载: 16 half/次, 两个累加器打断依赖 */
static void gemv_f16_256(float* y, const float* x, const unsigned short* W,
                         int in, int out) {
#pragma omp parallel for schedule(static)
    for (int j = 0; j < out; ++j) {
        const unsigned short* row = W + (size_t)j * in;
        __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
        int i = 0;
        for (; i + 16 <= in; i += 16) {
            __m256i h = _mm256_loadu_si256((const __m256i*)(row + i));
            __m256 w0 = _mm256_cvtph_ps(_mm256_castsi256_si128(h));
            __m256 w1 = _mm256_cvtph_ps(_mm256_extracti128_si256(h, 1));
            a0 = _mm256_fmadd_ps(w0, _mm256_loadu_ps(x + i), a0);
            a1 = _mm256_fmadd_ps(w1, _mm256_loadu_ps(x + i + 8), a1);
        }
        a0 = _mm256_add_ps(a0, a1);
        for (; i + 8 <= in; i += 8)
            a0 = _mm256_fmadd_ps(
                _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(row + i))),
                _mm256_loadu_ps(x + i), a0);
        float t[8];
        _mm256_storeu_ps(t, a0);
        float s = t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
        for (; i < in; ++i) s += (float)row[i] * x[i];
        y[j] = s;
    }
}

#define D 1408
#define NLAYER 8
#define VOCAB 8192
static const int SH[5][2] = {{1408, 1024}, {1024, 2816}, {1408, 1920},
                             {1920, 1408}, {1408, 1408}};

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    size_t per = 0;
    for (int k = 0; k < 5; ++k) per += (size_t)SH[k][0] * SH[k][1];
    const size_t total = per * NLAYER + (size_t)D * VOCAB;

    float* w32 = (float*)_aligned_malloc(total * 4, 64);
    unsigned short* w16 = (unsigned short*)_aligned_malloc(total * 2, 64);
    for (size_t i = 0; i < total; ++i) {
        float v = 0.001f * (float)(i % 89) - 0.04f;
        w32[i] = v;
        w16[i] = _cvtss_sh(v, 0);
    }
    const float* b32[NLAYER][5];
    const unsigned short* b16[NLAYER][5];
    {
        const float* p = w32; const unsigned short* q = w16;
        for (int l = 0; l < NLAYER; ++l)
            for (int k = 0; k < 5; ++k) {
                b32[l][k] = p; p += (size_t)SH[k][0] * SH[k][1];
                b16[l][k] = q; q += (size_t)SH[k][0] * SH[k][1];
            }
    }
    const float* h32 = w32 + per * NLAYER;
    const unsigned short* h16 = w16 + per * NLAYER;

    float xv[2816], yv[VOCAB];
    for (int i = 0; i < 2816; ++i) xv[i] = 0.3f;
    for (int i = 0; i < VOCAB; ++i) yv[i] = 0.3f;

    printf("fp32 权重 %.1f MB   fp16 %.1f MB\n\n", total * 4.0 / 1e6,
           total * 2.0 / 1e6);
    printf("%-5s %-16s %-16s %-16s %s\n", "线程", "fp32 ms", "f16-128 ms",
           "f16-256 ms", "256/32");

    for (int th = 1; th <= 8; th *= 2) {
        omp_set_num_threads(th);
        double t32 = 1e9, t128 = 1e9, t256 = 1e9;
        for (int r = 0; r < 10; ++r) {
            double t0 = now();
            for (int l = 0; l < NLAYER; ++l)
                for (int k = 0; k < 5; ++k) gemv_f32(yv, xv, b32[l][k], SH[k][0], SH[k][1]);
            gemv_f32(yv, xv, h32, D, VOCAB);
            double d = now() - t0; if (d < t32) t32 = d;

            t0 = now();
            for (int l = 0; l < NLAYER; ++l)
                for (int k = 0; k < 5; ++k) gemv_f16_128(yv, xv, b16[l][k], SH[k][0], SH[k][1]);
            gemv_f16_128(yv, xv, h16, D, VOCAB);
            d = now() - t0; if (d < t128) t128 = d;

            t0 = now();
            for (int l = 0; l < NLAYER; ++l)
                for (int k = 0; k < 5; ++k) gemv_f16_256(yv, xv, b16[l][k], SH[k][0], SH[k][1]);
            gemv_f16_256(yv, xv, h16, D, VOCAB);
            d = now() - t0; if (d < t256) t256 = d;
        }
        printf("%-5d %7.2f         %7.2f         %7.2f         %.2fx\n",
               th, t32 * 1e3, t128 * 1e3, t256 * 1e3, t32 / t256);
    }

    /* 数值一致性: 三种内核同源权重的输出偏差 */
    srand(11);
    for (int i = 0; i < 2816; ++i) xv[i] = (float)rand() / RAND_MAX - 0.5f;
    float ref[1024], r128[1024], r256[1024];
    gemv_f32(ref, xv, b32[0][0], SH[0][0], SH[0][1]);
    gemv_f16_128(r128, xv, b16[0][0], SH[0][0], SH[0][1]);
    gemv_f16_256(r256, xv, b16[0][0], SH[0][0], SH[0][1]);
    double d128 = 0, d256 = 0;
    for (int i = 0; i < 1024; ++i) {
        double a = fabs(r128[i] - ref[i]), b = fabs(r256[i] - ref[i]);
        if (a > d128) d128 = a;
        if (b > d256) d256 = b;
    }
    printf("\nvs fp32: 128位加载 max|diff|=%.3e   256位加载 max|diff|=%.3e\n", d128, d256);

    _aligned_free(w32); _aligned_free(w16);
    return 0;
}