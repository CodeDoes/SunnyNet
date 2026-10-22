/* 权重存储精度对单步解码的影响: fp32 vs fp16 (F16C 现场转换)。
 *
 * 逻辑上是访存受限, 每步必须读一遍全部权重。若把权重存成 fp16,
 * 每步读的字节数减半, 内核侧用 _mm256_cvtph_ps 转回 fp32 再做 FMA。
 * 本探针用同一份数值 (同源 fp32 截断成 fp16) 跑同一内核结构, 只比时间与带宽。
 *
 * 编译: gcc -O3 -mavx2 -mfma -march=x86-64-v3 -funroll-loops -fopenmp
 *       bench_fp16.c -o bench_fp16.exe
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

/* fp32 行主序: y[out] = W(out,in) · x(in) */
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

/* fp16 行主序: 读 8 个 half (16B) -> 转 8 个 float -> FMA。读字节减半。 */
static void gemv_f16(float* y, const float* x, const unsigned short* W, int in,
                     int out) {
#pragma omp parallel for schedule(static)
    for (int j = 0; j < out; ++j) {
        const unsigned short* row = W + (size_t)j * in;
        __m256 acc = _mm256_setzero_ps();
        int i = 0;
        for (; i + 8 <= in; i += 8) {
            __m128i h = _mm_loadu_si128((const __m128i*)(row + i));
            __m256 w = _mm256_cvtph_ps(h);
            acc = _mm256_fmadd_ps(w, _mm256_loadu_ps(x + i), acc);
        }
        float t[8];
        _mm256_storeu_ps(t, acc);
        float s = t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
        for (; i < in; ++i) s += (float)row[i] * x[i];
        y[j] = s;
    }
}

#define D 1408
#define NLAYER 8
static const int SH[5][2] = {{1408, 1024}, {1024, 2816}, {1408, 1920},
                             {1920, 1408}, {1408, 1408}};
#define VOCAB 8192

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);

    size_t per = 0;
    for (int k = 0; k < 5; ++k) per += (size_t)SH[k][0] * SH[k][1];
    const size_t total = per * NLAYER + (size_t)D * VOCAB;
    const double mb32 = total * 4.0 / 1e6;
    printf("权重 %zu 元素  fp32 %.1f MB  fp16 %.1f MB\n\n", total, mb32,
           total * 2.0 / 1e6);

    float* w32 = (float*)_aligned_malloc(total * 4, 64);
    unsigned short* w16 = (unsigned short*)_aligned_malloc(total * 2, 64);
    if (!w32 || !w16) { printf("alloc fail\n"); return 1; }
    for (size_t i = 0; i < total; ++i) {
        float v = 0.001f * (float)(i % 89) - 0.04f;
        w32[i] = v;
        w16[i] = _cvtss_sh(v, 0); /* round-to-nearest-even */
    }

    /* 逐块指针 */
    const float* b32[NLAYER][5];
    const unsigned short* b16[NLAYER][5];
    {
        const float* p = w32;
        const unsigned short* q = w16;
        for (int l = 0; l < NLAYER; ++l)
            for (int k = 0; k < 5; ++k) {
                b32[l][k] = p; p += (size_t)SH[k][0] * SH[k][1];
                b16[l][k] = q; q += (size_t)SH[k][0] * SH[k][1];
            }
    }
    const float* head32 = w32 + per * NLAYER;
    const unsigned short* head16 = w16 + per * NLAYER;

    float xv[2816], yv[VOCAB];
    for (int i = 0; i < 2816; ++i) xv[i] = 0.3f;
    for (int i = 0; i < VOCAB; ++i) yv[i] = 0.3f;

    printf("%-6s %-22s %-22s %s\n", "线程", "fp32 (ms / GB/s)", "fp16 (ms / GB/s)",
           "加速");
    for (int th = 1; th <= 8; th *= 2) {
        omp_set_num_threads(th);
        double b32t = 1e9, b16t = 1e9;
        for (int r = 0; r < 10; ++r) {
            double t0 = now();
            for (int l = 0; l < NLAYER; ++l)
                for (int k = 0; k < 5; ++k)
                    gemv_f32(yv, xv, b32[l][k], SH[k][0], SH[k][1]);
            gemv_f32(yv, xv, head32, D, VOCAB);
            double d = now() - t0;
            if (d < b32t) b32t = d;

            t0 = now();
            for (int l = 0; l < NLAYER; ++l)
                for (int k = 0; k < 5; ++k)
                    gemv_f16(yv, xv, b16[l][k], SH[k][0], SH[k][1]);
            gemv_f16(yv, xv, head16, D, VOCAB);
            d = now() - t0;
            if (d < b16t) b16t = d;
        }
        printf("%-6d %8.2f (%6.2f)        %8.2f (%6.2f)        %.2fx\n", th,
               b32t * 1e3, mb32 / b32t / 1e3, b16t * 1e3,
               mb32 / b16t / 1e3, /* fp16 有效带宽仍按 fp32 等价字节算 */
               b32t / b16t);
    }

    /* 数值: fp16 权重相对 fp32 的最大偏差 (随机 x) */
    srand(7);
    for (int i = 0; i < 2816; ++i) xv[i] = (float)rand() / RAND_MAX - 0.5f;
    for (int i = 0; i < 1024; ++i) yv[i] = 0.0f;
    gemv_f32(yv, xv, b32[0][0], SH[0][0], SH[0][1]);
    float ref[1024];
    memcpy(ref, yv, sizeof(float) * 1024);
    /*
      注意: b16 已在上面被反复使用, 这里重新从 w32 生成一份 fp16 参考。
      为公平, 用同样的源 (w32 第 0 层第 0 块)。
    */
    unsigned short* tmp16 = (unsigned short*)_aligned_malloc(
        (size_t)SH[0][0] * SH[0][1] * 2, 64);
    for (size_t i = 0; i < (size_t)SH[0][0] * SH[0][1]; ++i)
        tmp16[i] = _cvtss_sh(b32[0][0][i], 0);
    gemv_f16(yv, xv, tmp16, SH[0][0], SH[0][1]);
    float md = 0.0f;
    for (int i = 0; i < 1024; ++i) {
        float d = yv[i] - ref[i];
        if (d < 0) d = -d;
        if (d > md) md = d;
    }
    printf("\nVt 单块 fp16 vs fp32 输出 max|diff| = %.3e (量级 %.3f)\n", md,
           (double)md);
    _aligned_free(tmp16);

    _aligned_free(w32);
    _aligned_free(w16);
    return 0;
}