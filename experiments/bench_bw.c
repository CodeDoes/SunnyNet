/* 访存带宽探针: 分别测单线程与多线程的顺序读带宽。
 * 用于判断 SunnyNet C runtime 的单步解码是否还有多线程空间。
 * 编译: gcc -O3 -mavx2 -mfma -fopenmp bench_bw.c -o bench_bw.exe
 */
#include <immintrin.h>
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

/* 单线程 AVX2 顺序求和: 纯读, 不写 */
static double read_sum_1(const float* a, size_t n) {
    __m256 acc = _mm256_setzero_ps();
    for (size_t i = 0; i + 8 <= n; i += 8)
        acc = _mm256_add_ps(acc, _mm256_loadu_ps(a + i));
    float t[8];
    _mm256_storeu_ps(t, acc);
    double s = 0;
    for (int i = 0; i < 8; ++i) s += t[i];
    return s;
}

int main(void) {
    const size_t MB = 400;
    const size_t n = MB * 1024 * 1024 / 4;
    const size_t bytes = n * 4;
    float* a = (float*)malloc(bytes);
    if (!a) return 1;
    for (size_t i = 0; i < n; ++i) a[i] = 1.0f;
    printf("buffer %.0f MB\n\n", bytes / 1e6);

    const int reps = 5;
    for (int th = 1; th <= 8; th *= 2) {
        omp_set_num_threads(th);
        double best = 1e9;
        double sink = 0;
        for (int r = 0; r < reps; ++r) {
            double t0 = now();
            double s = 0;
#pragma omp parallel for reduction(+ : sink)
            for (int k = 0; k < th; ++k) {
                size_t chunk = n / th;
                size_t off = (size_t)k * chunk;
                sink += read_sum_1(a + off, chunk);
            }
            double dt = now() - t0;
            if (dt < best) best = dt;
        }
        (void)sink;
        printf("  %d 线程  顺序读  %6.2f GB/s  (%.1f ms/遍)\n",
               th, bytes / best / 1e9, best * 1e3);
    }
    free(a);
    return 0;
}