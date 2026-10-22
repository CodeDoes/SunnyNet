/* 单步解码瓶颈拆解: fork/join 开销、累加器依赖链、每 GEMV 独立并行 vs 整层单并行区。
 *
 * 权重规模照 p100mw (D=1408 I=1920 R=1024 K=4 L=8 vocab=8192),
 * 只关心访存, 数值无意义。
 * 编译: gcc -O3 -mavx2 -mfma -fopenmp bench_layer.c -o bench_layer.exe
 */
#include <immintrin.h>
#include <malloc.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* 单累加器版 (现状): acc 串行依赖, 每次迭代一个 FMA */
static void gemv1(float* y, const float* x, const float* W, int in, int out) {
#pragma omp for schedule(static) nowait
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

/* 4 累加器版: 打断依赖链, 每轮吃 32 个 float */
static void gemv4(float* y, const float* x, const float* W, int in, int out) {
#pragma omp for schedule(static) nowait
    for (int j = 0; j < out; ++j) {
        const float* row = W + (size_t)j * in;
        __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
        int i = 0;
        for (; i + 32 <= in; i += 32) {
            a0 = _mm256_fmadd_ps(_mm256_loadu_ps(row + i),
                                 _mm256_loadu_ps(x + i), a0);
            a1 = _mm256_fmadd_ps(_mm256_loadu_ps(row + i + 8),
                                 _mm256_loadu_ps(x + i + 8), a1);
            a2 = _mm256_fmadd_ps(_mm256_loadu_ps(row + i + 16),
                                 _mm256_loadu_ps(x + i + 16), a2);
            a3 = _mm256_fmadd_ps(_mm256_loadu_ps(row + i + 24),
                                 _mm256_loadu_ps(x + i + 24), a3);
        }
        a0 = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
        for (; i + 8 <= in; i += 8)
            a0 = _mm256_fmadd_ps(_mm256_loadu_ps(row + i),
                                 _mm256_loadu_ps(x + i), a0);
        float t[8];
        _mm256_storeu_ps(t, a0);
        float s = t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
        for (; i < in; ++i) s += row[i] * x[i];
        y[j] = s;
    }
}

static void gemv1_p(float* y, const float* x, const float* W, int in, int out) {
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

static void gemv4_p(float* y, const float* x, const float* W, int in, int out) {
#pragma omp parallel for schedule(static)
    for (int j = 0; j < out; ++j) {
        const float* row = W + (size_t)j * in;
        __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
        int i = 0;
        for (; i + 32 <= in; i += 32) {
            a0 = _mm256_fmadd_ps(_mm256_loadu_ps(row + i),
                                 _mm256_loadu_ps(x + i), a0);
            a1 = _mm256_fmadd_ps(_mm256_loadu_ps(row + i + 8),
                                 _mm256_loadu_ps(x + i + 8), a1);
            a2 = _mm256_fmadd_ps(_mm256_loadu_ps(row + i + 16),
                                 _mm256_loadu_ps(x + i + 16), a2);
            a3 = _mm256_fmadd_ps(_mm256_loadu_ps(row + i + 24),
                                 _mm256_loadu_ps(x + i + 24), a3);
        }
        a0 = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
        for (; i + 8 <= in; i += 8)
            a0 = _mm256_fmadd_ps(_mm256_loadu_ps(row + i),
                                 _mm256_loadu_ps(x + i), a0);
        float t[8];
        _mm256_storeu_ps(t, a0);
        float s = t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
        for (; i < in; ++i) s += row[i] * x[i];
        y[j] = s;
    }
}

#define D 1408
#define I 1920
#define R 1024
#define LAYERS 8
#define VOCAB 8192

/* 每层 5 个 GEMV 的 (in,out) */
static const int SH[5][2] = {
    {D, R}, {R, 2 * D}, {D, I}, {I, D}, {D, D}
};

static float* W[LAYERS][5];
static float* WHEAD;
/* 输出最大 VOCAB, 输入最大 D; 数值无意义, 只测访存 */
static float xbig[VOCAB], ybig[VOCAB];

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("配置 D=%d I=%d R=%d L=%d vocab=%d\n\n", D, I, R, LAYERS, VOCAB);

    /* fork/join 空开销 */
    for (int th = 1; th <= 8; th *= 2) {
        omp_set_num_threads(th);
        double best = 1e9;
        for (int r = 0; r < 200; ++r) {
            double t0 = now();
#pragma omp parallel for schedule(static)
            for (int j = 0; j < 64; ++j) ybig[j] = (float)j;
            double dt = now() - t0;
            if (dt < best) best = dt;
        }
        printf("空 parallel for (%2d 线程) %7.2f us\n", th, best * 1e6);
    }
    printf("\n");

    /* 分配: 8 层 × 5 块 + head, 总量 ≈ 421 MB */
    size_t total_elem = 0;
    for (int l = 0; l < LAYERS; ++l)
        for (int k = 0; k < 5; ++k) total_elem += (size_t)SH[k][0] * SH[k][1];
    total_elem += (size_t)D * VOCAB;
    float* big = (float*)_aligned_malloc(total_elem * 4 + 4096, 64);
    if (!big) { printf("alloc fail\n"); return 1; }
    printf("alloc ok: %.1f MB at %p\n\n", total_elem * 4.0 / 1e6, (void*)big);
    for (size_t i = 0; i < total_elem; ++i) big[i] = 0.001f * (float)(i % 89);
    const double gb = total_elem * 4.0 / 1e9;

    float* p = big;
    for (int l = 0; l < LAYERS; ++l)
        for (int k = 0; k < 5; ++k) { W[l][k] = p; p += (size_t)SH[k][0] * SH[k][1]; }
    WHEAD = p;

    for (int i = 0; i < VOCAB; ++i) { xbig[i] = 0.3f; ybig[i] = 0.3f; }

    const int reps = 8;

    /* 变体 A: 每 GEMV 独立 parallel for, 单累加器 (现状) */
    for (int th = 1; th <= 8; th *= 2) {
        omp_set_num_threads(th);
        double best = 1e9;
        for (int r = 0; r < reps; ++r) {
            double t0 = now();
            for (int l = 0; l < LAYERS; ++l) {
                gemv1_p(ybig, xbig, W[l][0], D, R);
                gemv1_p(ybig, xbig, W[l][1], R, 2 * D);
                gemv1_p(ybig, xbig, W[l][2], D, I);
                gemv1_p(ybig, xbig, W[l][3], I, D);
                gemv1_p(ybig, xbig, W[l][4], D, D);
            }
            gemv1_p(ybig, ybig, WHEAD, D, VOCAB);
            double dt = now() - t0;
            if (dt < best) best = dt;
        }
        printf("A 每GEMV独立并行 单累加器 %2d线程  %7.2f ms  %6.2f GB/s\n",
               th, best * 1e3, gb / best);
    }
    printf("\n");

    /* 变体 B: 每 GEMV 独立 parallel for, 4 累加器 */
    for (int th = 1; th <= 8; th *= 2) {
        omp_set_num_threads(th);
        double best = 1e9;
        for (int r = 0; r < reps; ++r) {
            double t0 = now();
            for (int l = 0; l < LAYERS; ++l) {
                gemv4_p(ybig, xbig, W[l][0], D, R);
                gemv4_p(ybig, xbig, W[l][1], R, 2 * D);
                gemv4_p(ybig, xbig, W[l][2], D, I);
                gemv4_p(ybig, xbig, W[l][3], I, D);
                gemv4_p(ybig, xbig, W[l][4], D, D);
            }
            gemv4_p(ybig, ybig, WHEAD, D, VOCAB);
            double dt = now() - t0;
            if (dt < best) best = dt;
        }
        printf("B 每GEMV独立并行 4累加器   %2d线程  %7.2f ms  %6.2f GB/s\n",
               th, best * 1e3, gb / best);
    }
    printf("\n");

    /* 变体 C: 整层一个 parallel 区, 内部 omp for nowait; 4 累加器 */
    for (int th = 1; th <= 8; th *= 2) {
        omp_set_num_threads(th);
        double best = 1e9;
        for (int r = 0; r < reps; ++r) {
            double t0 = now();
#pragma omp parallel
            {
                for (int l = 0; l < LAYERS; ++l) {
                    gemv4(ybig, xbig, W[l][0], D, R);
                    gemv4(ybig, xbig, W[l][1], R, 2 * D);
                    gemv4(ybig, xbig, W[l][2], D, I);
                    gemv4(ybig, xbig, W[l][3], I, D);
                    gemv4(ybig, xbig, W[l][4], D, D);
#pragma omp barrier
                }
                gemv4(ybig, ybig, WHEAD, D, VOCAB);
            }
            double dt = now() - t0;
            if (dt < best) best = dt;
        }
        printf("C 单并行区+nowait 4累加器 %2d线程  %7.2f ms  %6.2f GB/s\n",
               th, best * 1e3, gb / best);
    }
    printf("\n");

    /* 变体 D: 纯单线程顺序读基线 (同一块内存) */
    {
        omp_set_num_threads(1);
        double best = 1e9;
        for (int r = 0; r < reps; ++r) {
            double t0 = now();
            __m256 acc = _mm256_setzero_ps();
            for (size_t i = 0; i + 8 <= total_elem; i += 8)
                acc = _mm256_add_ps(acc, _mm256_loadu_ps(big + i));
            float t[8];
            _mm256_storeu_ps(t, acc);
            volatile float sink = t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
            (void)sink;
            double dt = now() - t0;
            if (dt < best) best = dt;
        }
        printf("D 单线程纯顺序读(同块)           %7.2f ms  %6.2f GB/s\n",
               best * 1e3, gb / best);
    }
    {
        omp_set_num_threads(4);
        double best = 1e9;
        for (int r = 0; r < reps; ++r) {
            double t0 = now();
            double sink = 0;
#pragma omp parallel for reduction(+ : sink) schedule(static)
            for (int k = 0; k < 4; ++k) {
                size_t chunk = total_elem / 4;
                const float* a = big + (size_t)k * chunk;
                __m256 acc = _mm256_setzero_ps();
                for (size_t i = 0; i + 8 <= chunk; i += 8)
                    acc = _mm256_add_ps(acc, _mm256_loadu_ps(a + i));
                float t[8];
                _mm256_storeu_ps(t, acc);
                sink += t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
            }
            double dt = now() - t0;
            if (dt < best) best = dt;
        }
        printf("D 4线程纯顺序读(同块)            %7.2f ms  %6.2f GB/s\n",
               best * 1e3, gb / best);
    }

    free(big);
    return 0;
}