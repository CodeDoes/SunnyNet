/* 直接调用真实 sunnynet.c 的 sn_step, 排除 Python/ctypes 开销。
 * 编译: gcc -O3 -mavx2 -mfma -fopenmp bench_real.c sunnynet.c -o bench_real.exe
 */
#include "sunnynet.h"

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    const char* blob = argv[1];
    int th = argc > 2 ? atoi(argv[2]) : 6;
    if (!blob) { printf("usage: bench_real <blob> [threads]\n"); return 2; }

    SnModel* m = sn_load(blob, 8192, 1408, 1920, 1024, 4, 8, 8, 1);
    if (!m) { printf("sn_load fail\n"); return 1; }
    printf("loaded. threads=%d\n", th);
    sn_set_threads(th);

    float* logits = (float*)malloc(sizeof(float) * 8192);
    const int NT = 32;
    int toks[NT];
    for (int i = 0; i < NT; ++i) toks[i] = 100 + (i * 37) % 4000;

    for (int r = 0; r < 3; ++r) { sn_reset(m); for (int t = 0; t < NT; ++t) sn_step(m, toks[t], logits); }

    /* 5 轮, 逐轮记录 */
    for (int r = 0; r < 5; ++r) {
        sn_reset(m);
        double t0 = now();
        for (int t = 0; t < NT; ++t) sn_step(m, toks[t], logits);
        double dt = now() - t0;
        printf("  round %d: %7.2f ms/step  (%6.2f GB/s)\n", r, dt / NT * 1e3,
               467.59e6 / (dt / NT) / 1e9);
    }
    free(logits);
    sn_free(m);
    return 0;
}