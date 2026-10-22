/* 对齐假设验证: 同权重、同内核, 只改起始地址的 64B 对齐, 看带宽差。
 * 编译: gcc -O3 -mavx2 -mfma -march=x86-64-v3 -funroll-loops -fopenmp
 *       bench_align.c -o bench_align.exe
 */
#include <immintrin.h>
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

/* 与 sunnynet.c 的 gemv 逐字一致 */
static void gemv(float* y, const float* x, const float* W, int in, int out) {
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

/* 层内 5 个 gemv 的 (in,out), 与 p100mw 一致 */
static const int SH[5][2] = {{1408, 1024}, {1024, 2816}, {1408, 1920}, {1920, 1408}, {1408, 1408}};
#define LAYERS 8
#define D 1408

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);

    /* 每层元素数(与真实 per_layer 一致, 含 delta=1408 与 scalar=2) */
    size_t per_layer = 0;
    for (int k = 0; k < 5; ++k) per_layer += (size_t)SH[k][0] * SH[k][1];
    per_layer += 2 * D /*ni*/ + 2 * 1408 /*nm 用 I=1920? 这里只做 gemv, 忽略*/;
    /* 用真实值: 2*1408 + 2*1920 + 4*1408 + delta 1408 + scalar 2 */
    per_layer = (size_t)5 * 0; /* reset */
    per_layer = (size_t)(2 * 1408 + 2 * 1920 + 4 * 1408 + 1408 + 2);
    for (int k = 0; k < 5; ++k) per_layer += (size_t)SH[k][0] * SH[k][1];
    printf("per_layer = %zu floats, bytes = %zu, /64 = %.3f  %s\n", per_layer,
           per_layer * 4, per_layer * 4.0 / 64,
           (per_layer * 4) % 64 == 0 ? "对齐" : "错位");

    const size_t total = per_layer * LAYERS;
    const double gb_all = total * 4.0 / 1e9;

    /* 两块: 紧凑连续 (现状) vs 每层头 64B 对齐 */
    float* tight = (float*)_aligned_malloc(total * 4, 64);
    float* alig = (float*)_aligned_malloc(total * 4 + LAYERS * 64, 64);
    if (!tight || !alig) { printf("alloc fail\n"); return 1; }
    for (size_t i = 0; i < total; ++i) tight[i] = 0.001f * (float)(i % 89);
    memset(alig, 0, total * 4 + LAYERS * 64);

    /* 定位每层 5 块的指针 */
    float* tw[LAYERS][5];
    float* aw[LAYERS][5];
    {
        float* p = tight;
        for (int l = 0; l < LAYERS; ++l) {
            p += 2 * 1408 + 2 * 1920 + 4 * 1408; /* ni/nm/dw 前置 */
            for (int k = 0; k < 5; ++k) { tw[l][k] = p; p += (size_t)SH[k][0] * SH[k][1]; }
            p += 1408 + 2;
        }
        /* 对齐版: 每层从 64B 边界开始, 块也各自对齐 */
        unsigned char* raw = (unsigned char*)alig;
        size_t off = 0;
        for (int l = 0; l < LAYERS; ++l) {
            off = (off + 63) & ~(size_t)63;
            float* q = (float*)(raw + off);
            for (int k = 0; k < 5; ++k) {
                off = (off + 63) & ~(size_t)63;
                q = (float*)(raw + off);
                aw[l][k] = q;
                memcpy(q, tw[l][k], (size_t)SH[k][0] * SH[k][1] * 4);
                off += (size_t)SH[k][0] * SH[k][1] * 4;
            }
        }
    }

    float xbig[2816], ybig[2816];
    for (int i = 0; i < 2816; ++i) { xbig[i] = 0.3f; ybig[i] = 0.3f; }

    printf("\n%-6s %-14s %-14s\n", "线程", "紧凑(现状) ms", "每块64对齐 ms");
    for (int th = 1; th <= 8; th *= 2) {
        omp_set_num_threads(th);
        double bt = 1e9, ba = 1e9;
        for (int r = 0; r < 10; ++r) {
            double t0 = now();
            for (int l = 0; l < LAYERS; ++l)
                for (int k = 0; k < 5; ++k)
                    gemv(ybig, xbig, tw[l][k], SH[k][0], SH[k][1]);
            double d = now() - t0;
            if (d < bt) bt = d;

            t0 = now();
            for (int l = 0; l < LAYERS; ++l)
                for (int k = 0; k < 5; ++k)
                    gemv(ybig, xbig, aw[l][k], SH[k][0], SH[k][1]);
            d = now() - t0;
            if (d < ba) ba = d;
        }
        printf("%-6d %8.2f (%5.1f) %8.2f (%5.1f)   %.2fx\n", th, bt * 1e3,
               gb_all / bt, ba * 1e3, gb_all / ba, bt / ba);
    }

    /* 打印错位量 */
    printf("\n紧凑布局下各层 Vt 地址 mod 64:\n");
    for (int l = 0; l < LAYERS; ++l)
        printf("  layer %d: ptr%%64 = %2d\n", l, (int)((uintptr_t)tw[l][0] % 64));

    _aligned_free(tight);
    _aligned_free(alig);
    return 0;
}