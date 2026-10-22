/* fp16 权重存储的端到端误差: 同源权重, fp32 与 fp16 两条前向路径,
 * 逐 token 比对 logits 与 argmax。
 *
 * 编译: gcc -O3 -mavx2 -mfma -march=x86-64-v3 -funroll-loops -fopenmp
 *       -I ../../sunnynet/runtime prof_fp16.c -o prof_fp16.exe
 */
#include "../../sunnynet/runtime/sunnynet.c"

#include <time.h>

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ---- fp16 权重的对应结构 ---- */
typedef struct {
    const unsigned short *Vt, *Ut, *Wz_t, *Wc_t, *Ws_t;
} SnLayer16;

typedef struct {
    int D, I, R, K, L, group, vocab, clamp;
    float eps, p_floor;
    SnLayer* lay;      /* fp32 层 (复用状态与偏置) */
    SnLayer16* lay16;
    const unsigned short* head16;
    const float* embed;
    const float* no_g;
    const float* no_c;
    float* sc;
    float* sc2;
} M16;

static void gemv_f16(float* y, const float* x, const unsigned short* W, int in,
                     int out) {
#pragma omp parallel for schedule(static)
    for (int j = 0; j < out; ++j) {
        const unsigned short* row = W + (size_t)j * in;
        __m256 acc = _mm256_setzero_ps();
        int i = 0;
        for (; i + 8 <= in; i += 8)
            acc = _mm256_fmadd_ps(_mm256_cvtph_ps(
                                      _mm_loadu_si128((const __m128i*)(row + i))),
                                  _mm256_loadu_ps(x + i), acc);
        float t[8];
        _mm256_storeu_ps(t, acc);
        float s = t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
        for (; i < in; ++i) s += (float)row[i] * x[i];
        y[j] = s;
    }
}

/* 与 layer_step 同结构, 只把 5 个 gemv 换成 fp16 权重版 */
static void layer_step16(M16* m, int li, const float* x, float* out) {
    SnLayer* L = &m->lay[li];
    SnLayer16* L6 = &m->lay16[li];
    const int D = m->D, I = m->I, R = m->R, K = m->K, G = m->group;
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

    layernorm(raw, x, L->ni_g, L->ni_c, D, m->eps);
    if (K > 1) {
        for (int i = 0; i < D; ++i) {
            float acc = raw[i] * L->dw_t[(size_t)(K - 1) * D + i];
            for (int k = 0; k < K - 1; ++k)
                acc += L->hist[(size_t)k * D + i] *
                       L->dw_t[(size_t)(K - 2 - k) * D + i];
            at[i] = raw[i] + acc;
        }
        for (int k = K - 2; k >= 1; --k)
            memcpy(L->hist + (size_t)k * D, L->hist + (size_t)(k - 1) * D,
                   sizeof(float) * D);
        memcpy(L->hist, raw, sizeof(float) * D);
    } else {
        memcpy(at, raw, sizeof(float) * D);
    }

    gemv_f16(v, at, L6->Vt, D, R);
    gemv_f16(q, v, L6->Ut, R, 2 * D);
    memcpy(u, q, sizeof(float) * D);
    memcpy(gate, q + D, sizeof(float) * D);
    for (int i = 0; i < D; ++i) {
        const int base = (i / G) * G, off = i - base;
        u[i] *= sgn(gate[i] - gate[base + (off + 1) % G]) *
                sgn(gate[i] - gate[base + (off + 2) % G]);
    }
    gemv_f16(sfb, L->s, L6->Ws_t, D, D);
    for (int i = 0; i < D; ++i) {
        float p = 1.0f - L->delta[i] * (1.0f - sgn(gate[i]));
        if (m->clamp && p < m->p_floor) p = m->p_floor;
        L->s[i] = p * L->s[i] + (1.0f - p) * u[i];
    }
    gemv_f16(zn, u, L6->Wz_t, D, I);
    for (int i = 0; i < I; ++i)
        zn[i] *= 1.0f + 0.5f * L->a_tanh / (1.0f + fabsf(zn[i]));
    layernorm(yn, zn, L->nm_g, L->nm_c, I, m->eps);
    gemv_f16(out, yn, L6->Wc_t, I, D);
    for (int i = 0; i < D; ++i) out[i] += L->b_gain * L->s[i] + sfb[i] + x[i];
}

int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    const char* blob = argc > 1 ? argv[1]
                                : "dist/release/sunnynet-p100mw.runtime.bin";

    SnModel* m = sn_load(blob, 8192, 1408, 1920, 1024, 4, 8, 8, 1);
    if (!m) { printf("sn_load fail\n"); return 1; }

    const int D = m->D, I = m->I, R = m->R, K = m->K, L = m->L;
    printf("模型 D=%d I=%d R=%d K=%d L=%d vocab=%d\n", D, I, R, K, L, m->vocab);

    /* 构造 fp16 副本 */
    M16 mm;
    mm.D = D; mm.I = I; mm.R = R; mm.K = K; mm.L = L; mm.group = m->group;
    mm.vocab = m->vocab; mm.eps = m->eps; mm.p_floor = m->p_floor;
    mm.clamp = m->clamp;
    mm.lay = m->layers;
    mm.embed = m->embed; mm.no_g = m->no_g; mm.no_c = m->no_c;
    mm.lay16 = (SnLayer16*)calloc((size_t)L, sizeof(SnLayer16));
    mm.sc = m->sc;
    mm.sc2 = m->sc2;

    size_t tot16 = 0;
    for (int l = 0; l < L; ++l) {
        SnLayer* Lp = &m->layers[l];
        SnLayer16* L6 = &mm.lay16[l];
        size_t nv = (size_t)D * R, nu = (size_t)R * 2 * D, nz = (size_t)D * I;
        size_t nc = (size_t)I * D, ns = (size_t)D * D;
        unsigned short* buf = (unsigned short*)malloc(
            sizeof(unsigned short) * (nv + nu + nz + nc + ns));
        unsigned short* p = buf;
        const float* src[5] = {Lp->Vt, Lp->Ut, Lp->Wz_t, Lp->Wc_t, Lp->Ws_t};
        size_t cnt[5] = {nv, nu, nz, nc, ns};
        const unsigned short* dst[5];
        for (int k = 0; k < 5; ++k) {
            dst[k] = p;
            for (size_t i = 0; i < cnt[k]; ++i) p[i] = _cvtss_sh(src[k][i], 0);
            p += cnt[k];
        }
        L6->Vt = dst[0]; L6->Ut = dst[1]; L6->Wz_t = dst[2];
        L6->Wc_t = dst[3]; L6->Ws_t = dst[4];
        tot16 += nv + nu + nz + nc + ns;
    }
    {
        size_t nh = (size_t)D * m->vocab;
        unsigned short* h = (unsigned short*)malloc(sizeof(unsigned short) * nh);
        for (size_t i = 0; i < nh; ++i) h[i] = _cvtss_sh(m->head_t[i], 0);
        mm.head16 = h;
        tot16 += nh;
    }
    printf("fp16 权重 %.1f MB (fp32 %.1f MB)\n\n", tot16 * 2.0 / 1e6,
           467.59);

    /* fp32 参考: 用真实 sn_step */
    float* lo32 = (float*)malloc(sizeof(float) * m->vocab);
    float* lo16 = (float*)malloc(sizeof(float) * m->vocab);
    float* x = m->sc2;
    float* lout = x + D;
    float* nrm = lout + D;

    const int NT = 48;
    int toks[NT];
    srand(1234);
    toks[0] = 1;
    for (int i = 1; i < NT; ++i) toks[i] = rand() % 8000;

    /* ---- 误差 ---- */
    sn_reset(m);
    float* ref_all = (float*)malloc(sizeof(float) * m->vocab * NT);
    int* arg32 = (int*)malloc(sizeof(int) * NT);
    for (int t = 0; t < NT; ++t) {
        sn_step(m, toks[t], lo32);
        memcpy(ref_all + (size_t)t * m->vocab, lo32, sizeof(float) * m->vocab);
        int best = 0;
        for (int v = 1; v < m->vocab; ++v)
            if (lo32[v] > lo32[best]) best = v;
        arg32[t] = best;
    }

    sn_reset(m);
    double worst = 0.0;
    int agree = 0;
    double rel_worst = 0.0;
    for (int t = 0; t < NT; ++t) {
        memcpy(x, m->embed + (size_t)toks[t] * D, sizeof(float) * D);
        for (int l = 0; l < L; ++l) {
            layer_step16(&mm, l, x, lout);
            memcpy(x, lout, sizeof(float) * D);
        }
        layernorm(nrm, x, m->no_g, m->no_c, D, m->eps);
        gemv_f16(lo16, nrm, mm.head16, D, m->vocab);
        const float* r = ref_all + (size_t)t * m->vocab;
        double mxd = 0.0, scale = 0.0;
        for (int v = 0; v < m->vocab; ++v) {
            double d = fabs((double)lo16[v] - r[v]);
            if (d > mxd) mxd = d;
            if (fabs((double)r[v]) > scale) scale = fabs((double)r[v]);
        }
        if (mxd > worst) worst = mxd;
        if (scale > 0 && mxd / scale > rel_worst) rel_worst = mxd / scale;
        int best = 0;
        for (int v = 1; v < m->vocab; ++v)
            if (lo16[v] > lo16[best]) best = v;
        if (best == arg32[t]) ++agree;
    }
    printf("=== fp16 权重 vs fp32 权重 (同一 token 序列, %d 步) ===\n", NT);
    printf("  logits max|diff| 最坏        %.3e\n", worst);
    printf("  相对 max|diff|/max|logit|    %.3e\n", rel_worst);
    printf("  argmax 一致                  %d/%d\n", agree, NT);

    /* ---- 速度 ---- */
    for (int th = 1; th <= 8; th *= 2) {
        sn_set_threads(th);
        /* fp32 路径 */
        double b32 = 1e9;
        for (int r = 0; r < 5; ++r) {
            sn_reset(m);
            double t0 = now();
            for (int t = 0; t < NT; ++t) sn_step(m, toks[t], lo32);
            double d = (now() - t0) / NT;
            if (d < b32) b32 = d;
        }
        /* fp16 路径 */
        double b16 = 1e9;
        for (int r = 0; r < 5; ++r) {
            sn_reset(m);
            double t0 = now();
            for (int t = 0; t < NT; ++t) {
                memcpy(x, m->embed + (size_t)toks[t] * D, sizeof(float) * D);
                for (int l = 0; l < L; ++l) {
                    layer_step16(&mm, l, x, lout);
                    memcpy(x, lout, sizeof(float) * D);
                }
                layernorm(nrm, x, m->no_g, m->no_c, D, m->eps);
                gemv_f16(lo16, nrm, mm.head16, D, m->vocab);
            }
            double d = (now() - t0) / NT;
            if (d < b16) b16 = d;
        }
        printf("  %d 线程: fp32 %6.2f ms  fp16 %6.2f ms   %.2fx\n", th,
               b32 * 1e3, b16 * 1e3, b32 / b16);
    }

    free(lo32); free(lo16); free(ref_all); free(arg32);
    for (int l = 0; l < L; ++l) free((void*)mm.lay16[l].Vt);
    free(mm.lay16); free((void*)mm.head16);
    sn_free(m);
    return 0;
}