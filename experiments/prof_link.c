/* 逐段计时: 直接 include 真实 sunnynet.c, 从而访问其 static gemv/layer_step。
 * 编译: gcc -O3 -mavx2 -mfma -march=x86-64-v3 -funroll-loops -fopenmp
 *       -I ../../sunnynet/runtime prof_link.c -o prof_link.exe
 */
#include "../../sunnynet/runtime/sunnynet.c"

#include <time.h>

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* 累加器: 每段耗时 */
static double T_gemv = 0, T_taps = 0, T_compete = 0, T_sfb = 0, T_state = 0;
static double T_zn = 0, T_ln = 0, T_add = 0;

/* gemv 计时包装: 直接调用真实 gemv */
static void gemv_timed(float* y, const float* x, const float* W, int in, int out) {
    double t0 = now();
    gemv(y, x, W, in, out);
    T_gemv += now() - t0;
}

/* 复刻 layer_step, 但把每段拆开计时; 数值与真实实现一致 */
static void layer_step_prof(SnModel* m, SnLayer* L, const float* x, float* out) {
    const int D = L->D, I = L->I, R = L->R, K = L->K, G = m->group;
    float* sc = m->sc;
    float* raw = sc;
    float* at = raw + D;
    float* v = at + D;
    float* q = v + R;
    float* u = q + 2 * D;
    float* gate = u + D;
    float* zn = gate + D;
    float* yn = zn + I;
    float* sfb = yn + I;

    double t0 = now();
    layernorm(raw, x, L->ni_g, L->ni_c, D, m->eps);
    T_ln += now() - t0;

    t0 = now();
    if (K > 1) {
        for (int i = 0; i < D; ++i) {
            float acc = raw[i] * L->dw_t[(size_t)(K - 1) * D + i];
            for (int k = 0; k < K - 1; ++k)
                acc += L->hist[(size_t)k * D + i] * L->dw_t[(size_t)(K - 2 - k) * D + i];
            at[i] = raw[i] + acc;
        }
        for (int k = K - 2; k >= 1; --k)
            memcpy(L->hist + (size_t)k * D, L->hist + (size_t)(k - 1) * D,
                   sizeof(float) * D);
        memcpy(L->hist, raw, sizeof(float) * D);
    } else {
        memcpy(at, raw, sizeof(float) * D);
    }
    T_taps += now() - t0;

    gemv_timed(v, at, L->Vt, D, R);
    gemv_timed(q, v, L->Ut, R, 2 * D);
    memcpy(u, q, sizeof(float) * D);
    memcpy(gate, q + D, sizeof(float) * D);

    t0 = now();
    for (int i = 0; i < D; ++i) {
        const int base = (i / G) * G, off = i - base;
        u[i] *= sgn(gate[i] - gate[base + (off + 1) % G]) *
                sgn(gate[i] - gate[base + (off + 2) % G]);
    }
    T_compete += now() - t0;

    t0 = now();
    gemv(sfb, L->s, L->Ws_t, D, D);
    T_sfb += now() - t0;

    t0 = now();
    for (int i = 0; i < D; ++i) {
        float p = 1.0f - L->delta[i] * (1.0f - sgn(gate[i]));
        if (m->clamp && p < m->p_floor) p = m->p_floor;
        L->s[i] = p * L->s[i] + (1.0f - p) * u[i];
    }
    T_state += now() - t0;

    gemv_timed(zn, u, L->Wz_t, D, I);

    t0 = now();
    for (int i = 0; i < I; ++i)
        zn[i] *= 1.0f + 0.5f * L->a_tanh / (1.0f + fabsf(zn[i]));
    layernorm(yn, zn, L->nm_g, L->nm_c, I, m->eps);
    T_zn += now() - t0;

    gemv_timed(out, yn, L->Wc_t, I, D);

    t0 = now();
    for (int i = 0; i < D; ++i) out[i] += L->b_gain * L->s[i] + sfb[i] + x[i];
    T_add += now() - t0;
}

int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    const char* blob = argc > 1 ? argv[1]
                                : "dist/release/sunnynet-p100mw.runtime.bin";
    int th = argc > 2 ? atoi(argv[2]) : 6;

    SnModel* m = sn_load(blob, 8192, 1408, 1920, 1024, 4, 8, 8, 1);
    if (!m) { printf("sn_load fail\n"); return 1; }
    sn_set_threads(th);
    printf("threads=%d\n", th);

    const int D = m->D, L = m->L;
    float* x = m->sc2;
    float* lout = x + D;
    float* nrm = lout + D;
    float* logits = (float*)malloc(sizeof(float) * m->vocab);
    int toks[32];
    for (int i = 0; i < 32; ++i) toks[i] = 100 + (i * 37) % 4000;

    /* warmup */
    sn_reset(m);
    for (int t = 0; t < 32; ++t) sn_step(m, toks[t], logits);

    /* 纯 gemv 循环: 同样 41 次调用、同样权重指针, 但无串行依赖段。
     * 与下面的会话结构对比, 差异即为串行段/依赖造成的损失。 */
    {
        float* gx = (float*)malloc(sizeof(float) * 2816);
        float* gy = (float*)malloc(sizeof(float) * 2816);
        float* gl = (float*)malloc(sizeof(float) * 8192);
        for (int i = 0; i < 2816; ++i) { gx[i] = 0.3f; gy[i] = 0.3f; }
        for (int i = 0; i < 8192; ++i) gl[i] = 0.3f;
        double best = 1e9;
        for (int r = 0; r < 10; ++r) {
            double a = now();
            for (int l = 0; l < L; ++l) {
                SnLayer* Lp = &m->layers[l];
                gemv(gy, gx, Lp->Vt, Lp->D, Lp->R);
                gemv(gy, gx, Lp->Ut, Lp->R, 2 * Lp->D);
                gemv(gy, gx, Lp->Wz_t, Lp->D, Lp->I);
                gemv(gy, gx, Lp->Wc_t, Lp->I, Lp->D);
                gemv(gy, gx, Lp->Ws_t, Lp->D, Lp->D);
            }
            gemv(gl, gx, m->head_t, m->D, m->vocab);
            double d = now() - a;
            if (d < best) best = d;
        }
        printf("纯 gemv 循环(41 次)      %7.2f ms  %6.2f GB/s\n", best * 1e3,
               467.59e6 / best / 1e9);
        free(gx); free(gy); free(gl);
    }

    /* 计时: 与真实 sn_step 同结构; 逐步记录, 报 min/median, 避免用手动平均值 */
    const int REPS = 8, NS = 16;
    T_gemv = T_taps = T_compete = T_sfb = T_state = T_zn = T_ln = T_add = 0;
    double step_ms[REPS * NS];
    int ns = 0;
    for (int r = 0; r < REPS; ++r) {
        sn_reset(m);
        for (int s = 0; s < NS; ++s) {
            int tok = toks[s];
            memcpy(x, m->embed + (size_t)tok * D, sizeof(float) * D);
            double t0 = now();
            for (int l = 0; l < L; ++l) {
                layer_step_prof(m, &m->layers[l], x, lout);
                memcpy(x, lout, sizeof(float) * D);
            }
            layernorm(nrm, x, m->no_g, m->no_c, D, m->eps);
            gemv_timed(logits, nrm, m->head_t, D, m->vocab);
            step_ms[ns++] = (now() - t0) * 1e3;
        }
    }
    /* 排序取中位与最小 */
    for (int i = 0; i < ns; ++i)
        for (int j = i + 1; j < ns; ++j)
            if (step_ms[j] < step_ms[i]) {
                double t = step_ms[i]; step_ms[i] = step_ms[j]; step_ms[j] = t;
            }
    const double per = step_ms[0];
    const double med = step_ms[ns / 2];
    double total = 0;
    for (int i = 0; i < ns; ++i) total += step_ms[i];

    printf("\n总计 min %.2f / med %.2f / mean %.2f ms/step (%d 步)\n", per, med,
           total / ns, ns);
    printf("  gemv 段(40 次)   %7.2f ms  %5.1f%%\n", T_gemv / ns * 1e3,
           100 * T_gemv / (total / 1e3));
    printf("  sfb gemv         %7.3f ms  %5.1f%%\n", T_sfb / ns * 1e3,
           100 * T_sfb / (total / 1e3));
    printf("  compete(sigmoid) %7.3f ms  %5.1f%%\n", T_compete / ns * 1e3,
           100 * T_compete / (total / 1e3));
    printf("  state 更新       %7.3f ms  %5.1f%%\n", T_state / ns * 1e3,
           100 * T_state / (total / 1e3));
    printf("  taps 卷积        %7.3f ms  %5.1f%%\n", T_taps / ns * 1e3,
           100 * T_taps / (total / 1e3));
    printf("  zn+layernorm     %7.3f ms  %5.1f%%\n", T_zn / ns * 1e3,
           100 * T_zn / (total / 1e3));
    printf("  输入 layernorm   %7.3f ms  %5.1f%%\n", T_ln / ns * 1e3,
           100 * T_ln / (total / 1e3));
    printf("  输出相加         %7.3f ms  %5.1f%%\n", T_add / ns * 1e3,
           100 * T_add / (total / 1e3));
    printf("  (gemv 合计 %.2f ms, 有效 %.2f GB/s)\n",
           (T_gemv + T_sfb) / ns * 1e3, 467.59e6 / ((T_gemv + T_sfb) / ns) / 1e9);

    free(logits);
    sn_free(m);
    return 0;
}