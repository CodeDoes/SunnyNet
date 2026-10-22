/* 验证 sn_load_h: 与 sn_load (fp32) 逐 token 比对 logits 与 argmax, 并测速。
 * 编译: gcc -O3 -mavx2 -mfma -march=x86-64-v3 -funroll-loops -fopenmp
 *       -I sunnynet/runtime experiments/verify_h.c -o experiments/verify_h.exe
 */
#include "../../sunnynet/runtime/sunnynet.c"

#include <math.h>
#include <time.h>

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    const char* blob = argc > 1 ? argv[1]
                                : "dist/release/sunnynet-p100mw.runtime.bin";
    const int V = 8192, D = 1408, I = 1920, R = 1024, K = 4, L = 8, G = 8;

    SnModel* a = sn_load(blob, V, D, I, R, K, L, G, 1);
    SnModel* b = sn_load_h(blob, V, D, I, R, K, L, G, 1);
    if (!a || !b) { printf("load fail fp32=%p fp16=%p\n", (void*)a, (void*)b); return 1; }
    printf("both loaded (fp32 && fp16)\n");

    float* la = (float*)malloc(sizeof(float) * V);
    float* lb = (float*)malloc(sizeof(float) * V);
    const int NT = 64;
    int toks[NT];
    srand(2026);
    toks[0] = 1;
    for (int i = 1; i < NT; ++i) toks[i] = rand() % 8000;

    /* 逐 token 误差 */
    sn_reset(a); sn_reset(b);
    double worst = 0, relw = 0;
    int agree = 0;
    for (int t = 0; t < NT; ++t) {
        sn_step(a, toks[t], la);
        sn_step(b, toks[t], lb);
        double md = 0, sc = 0;
        int ba = 0, bb = 0;
        for (int v = 0; v < V; ++v) {
            double d = fabs((double)lb[v] - la[v]);
            if (d > md) md = d;
            if (fabs((double)la[v]) > sc) sc = fabs((double)la[v]);
            if (la[v] > la[ba]) ba = v;
            if (lb[v] > lb[bb]) bb = v;
        }
        if (md > worst) worst = md;
        if (sc > 0 && md / sc > relw) relw = md / sc;
        if (ba == bb) ++agree;
    }
    printf("\n=== fp16 权重 vs fp32 权重, %d 步 ===\n", NT);
    printf("  logits max|diff| 最坏   %.3e\n", worst);
    printf("  相对偏差                %.3e\n", relw);
    printf("  argmax 一致             %d/%d\n", agree, NT);

    /* 状态一致性: 同样步数后各层 s 的偏差 */
    {
        double smax = 0;
        for (int l = 0; l < L; ++l) {
            SnLayer* La = &a->layers[l];
            SnLayer* Lb = &b->layers[l];
            for (int i = 0; i < D; ++i) {
                double d = fabs((double)Lb->s[i] - La->s[i]);
                if (d > smax) smax = d;
            }
        }
        printf("  末态 state max|diff|    %.3e\n", smax);
    }

    /* 速度 */
    printf("\n线程   fp32 ms   fp16 ms   加速\n");
    for (int th = 1; th <= 8; th *= 2) {
        sn_set_threads(th);
        double ta = 1e9, tb = 1e9;
        for (int r = 0; r < 6; ++r) {
            sn_reset(a);
            double t0 = now();
            for (int t = 0; t < NT; ++t) sn_step(a, toks[t], la);
            double d = (now() - t0) / NT;
            if (d < ta) ta = d;

            sn_reset(b);
            t0 = now();
            for (int t = 0; t < NT; ++t) sn_step(b, toks[t], lb);
            d = (now() - t0) / NT;
            if (d < tb) tb = d;
        }
        printf("%4d  %8.2f  %8.2f   %.2fx\n", th, ta * 1e3, tb * 1e3, ta / tb);
    }

    free(la); free(lb);
    sn_free(a); sn_free(b);
    return 0;
}