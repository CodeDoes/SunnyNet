/* GEMV 实现对比: 当前 stride 布局 (单线程) vs 分块 + OpenMP。
 * 权重布局 (in, out) 行主序, y = W^T x 即 y[out] = sum_in x[in]*W[in*out+j]。
 * 编译: gcc -O3 -mavx2 -mfma -fopenmp -static bench_gemv.c -o bench_gemv.exe
 */
#include <immintrin.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* 当前 runtime 的实现: 外层输出块, 内层遍历 in, 每次 loadu 8 个 float */
static void gemv_cur(float* y, const float* x, const float* Wt, int in, int out) {
    int j = 0;
    for (; j + 8 <= out; j += 8) {
        __m256 acc = _mm256_setzero_ps();
        const float* row = Wt + j;
        for (int i = 0; i < in; ++i) {
            const __m256 b = _mm256_set1_ps(x[i]);
            acc = _mm256_fmadd_ps(b, _mm256_loadu_ps(row + (size_t)i * out), acc);
        }
        _mm256_storeu_ps(y + j, acc);
    }
    for (; j < out; ++j) {
        float acc = 0.0f;
        const float* row = Wt + j;
        for (int i = 0; i < in; ++i) acc += x[i] * row[(size_t)i * out];
        y[j] = acc;
    }
}

/* 并行版: 每个线程负责一段输出通道, 内层同样是 stride 遍历 */
static void gemv_par(float* y, const float* x, const float* Wt, int in, int out,
                     int threads) {
    omp_set_num_threads(threads);
#pragma omp parallel for schedule(static)
    for (int j0 = 0; j0 < out; j0 += 16) {
        int jn = (j0 + 16 <= out) ? 16 : (out - j0);
        int j = j0;
        for (; j + 8 <= j0 + jn; j += 8) {
            __m256 acc = _mm256_setzero_ps();
            const float* row = Wt + j;
            for (int i = 0; i < in; ++i) {
                const __m256 b = _mm256_set1_ps(x[i]);
                acc = _mm256_fmadd_ps(b, _mm256_loadu_ps(row + (size_t)i * out), acc);
            }
            _mm256_storeu_ps(y + j, acc);
        }
        for (; j < j0 + jn; ++j) {
            float acc = 0.0f;
            const float* row = Wt + j;
            for (int i = 0; i < in; ++i) acc += x[i] * row[(size_t)i * out];
            y[j] = acc;
        }
    }
}

/* 转置布局 (out, in) 行主序: 内层连续读, 天然向量化与预取友好 */
static void gemv_rowmajor(float* y, const float* x, const float* W, int in,
                          int out, int threads) {
    omp_set_num_threads(threads);
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

static double bench(void (*fn)(float*, const float*, const float*, int, int),
                    float* y, const float* x, const float* W, int in, int out,
                    int reps) {
    fn(y, x, W, in, out); /* warm */
    double best = 1e9;
    for (int r = 0; r < reps; ++r) {
        double t0 = now();
        fn(y, x, W, in, out);
        double dt = now() - t0;
        if (dt < best) best = dt;
    }
    return best;
}

int main(void) {
    /* 用 p100mw 的最大单次 GEMV (输出投影 D->vocab) 与全模型权重规模 */
    struct { const char* name; int in, out; } cases[] = {
        {"head   D->vocab", 1408, 8192},
        {"Wz     D->I    ", 1408, 1920},
        {"Ws     D->D    ", 1408, 1408},
    };
    const int ncase = (int)(sizeof(cases) / sizeof(cases[0]));

    for (int c = 0; c < ncase; ++c) {
        int in = cases[c].in, out = cases[c].out;
        size_t wn = (size_t)in * out;
        float* Wt = (float*)malloc(wn * 4);
        float* Wr = (float*)malloc(wn * 4);
        float* x = (float*)malloc((size_t)in * 4);
        float* y = (float*)malloc((size_t)out * 4);
        for (size_t i = 0; i < wn; ++i) Wt[i] = 0.01f * (float)(i % 97);
        for (int i = 0; i < in; ++i) x[i] = 0.5f;
        /* 转置: Wr[j*in+i] = Wt[i*out+j] */
        for (int j = 0; j < out; ++j)
            for (int i = 0; i < in; ++i) Wr[(size_t)j * in + i] = Wt[(size_t)i * out + j];

        double mb = wn * 4 / 1e6;
        printf("%s  in=%d out=%d  权重 %.1f MB\n", cases[c].name, in, out, mb);
        double t1 = bench(gemv_cur, y, x, Wt, in, out, 20);
        printf("  当前 stride 1线程   %7.2f ms  %6.2f GB/s\n", t1 * 1e3, mb / t1 / 1e3);

        for (int th = 4; th <= 8; th *= 2) {
            double best = 1e9;
            for (int r = 0; r < 20; ++r) {
                double a = now();
                gemv_par(y, x, Wt, in, out, th);
                double d = now() - a;
                if (d < best) best = d;
            }
            printf("  并行 stride %d线程   %7.2f ms  %6.2f GB/s  (%.2fx)\n",
                   th, best * 1e3, mb / best / 1e3, t1 / best);
        }
        for (int th = 1; th <= 8; th *= 2) {
            double best = 1e9;
            for (int r = 0; r < 20; ++r) {
                double a = now();
                gemv_rowmajor(y, x, Wr, in, out, th);
                double d = now() - a;
                if (d < best) best = d;
            }
            printf("  行主序 %d线程        %7.2f ms  %6.2f GB/s  (%.2fx)\n",
                   th, best * 1e3, mb / best / 1e3, t1 / best);
        }
        printf("\n");
        free(Wt); free(Wr); free(x); free(y);
    }
    return 0;
}