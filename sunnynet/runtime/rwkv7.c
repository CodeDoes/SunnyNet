/* SunnyNet Runtime: RWKV-7 单步解码, 整模型在 C (AVX2/FMA).
 *
 * 权重矩阵一律按 gemv 约定存 (in,out) 行主序: y = Wt^T x.
 * 状态 S 逐 head (HS,HS); rank-1 校正项与衰减项共用更新前的 S.
 * w 的 LoRA 中间夹 tanh, 不可折叠; a/v 可折但为统一起见也保留分解.
 * 单线程: B=1 单序列, 张量太小, 起线程的唤醒开销大于收益.
 */
#include "rwkv7.h"

#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
/* mingw 的 fopen 走 ANSI 代码页, 非 ASCII 路径打不开. */
static FILE* sn_fopen(const char* path, const char* mode) {
    wchar_t wp[1024], wm[16];
    if (MultiByteToWideChar(CP_UTF8, 0, path, -1, wp, 1024) <= 0) return NULL;
    MultiByteToWideChar(CP_UTF8, 0, mode, -1, wm, 16);
    return _wfopen(wp, wm);
}
#else
#define sn_fopen fopen
#endif

#define SN_RK_MAGIC 0x4B57524B /* "KRWK" */
#define SN_RK_VERSION 2
#define SN_R2 0.6065306597126334f /* 1/sqrt(e) */
#define SN_EPS_GROUP 6.4e-4f

struct SnRWKV7;

typedef struct {
    int D, HS, NH, F, Rw, Ra, Rv, Rg;
    const float *ln1w, *ln1b, *ln2w, *ln2b;
    const float* lerp;              /* (6,D) */
    const float *rw, *kw, *vw, *ow; /* (D,D) */
    const float *w1, *w2, *a1, *a2, *v1, *v2, *g1, *g2;
    const float *w0, *a0, *v0;
    const float *kk_w, *ka_w, *rk_w;
    const float *lnx_w, *lnx_b;
    const float* cmk;               /* (D,F) */
    const float* cmv;               /* (F,D) */
    const float* cmk_lerp;
    float* S;                       /* (NH,HS,HS) */
    float* shift;                   /* (2,D) */
    int first;                      /* layer 0: 记录 v_first 而不混合 */
    struct SnRWKV7* m;
} SnLayer;

struct SnRWKV7 {
    int vocab, D, HS, NH, F, L;
    float eps, eps_group;
    float* wbuf;
    SnLayer* layers;
    const float *en_w, *en_b, *on_w, *on_b;
    const float* embed;   /* (vocab,D) */
    const float* head_t;  /* (D,vocab) */
    float* sc;            /* 逐层工作区 */
    float* top;           /* (3D): x, nrm, vfirst */
    int ready;
};

static inline float hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    return _mm_cvtss_f32(lo);
}

static inline float sgn(float x) { return 1.0f / (1.0f + expf(-x)); }

/* 处理输出列区间 [j0, j0+jn), Wt 存 (in,out). */
static void gemv_blk(float* y, const float* x, const float* Wt, int in, int out,
                     int j0, int jn) {
    const int end = j0 + jn;
    int j = j0;
    for (; j + 32 <= end; j += 32) {
        __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
        __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
        for (int i = 0; i < in; ++i) {
            const __m256 b = _mm256_set1_ps(x[i]);
            const float* row = Wt + (size_t)i * out + j;
            a0 = _mm256_fmadd_ps(b, _mm256_loadu_ps(row), a0);
            a1 = _mm256_fmadd_ps(b, _mm256_loadu_ps(row + 8), a1);
            a2 = _mm256_fmadd_ps(b, _mm256_loadu_ps(row + 16), a2);
            a3 = _mm256_fmadd_ps(b, _mm256_loadu_ps(row + 24), a3);
        }
        _mm256_storeu_ps(y + j, a0);
        _mm256_storeu_ps(y + j + 8, a1);
        _mm256_storeu_ps(y + j + 16, a2);
        _mm256_storeu_ps(y + j + 24, a3);
    }
    for (; j + 8 <= end; j += 8) {
        __m256 acc = _mm256_setzero_ps();
        for (int i = 0; i < in; ++i) {
            const __m256 b = _mm256_set1_ps(x[i]);
            acc = _mm256_fmadd_ps(b, _mm256_loadu_ps(Wt + (size_t)i * out + j), acc);
        }
        _mm256_storeu_ps(y + j, acc);
    }
    for (; j < end; ++j) {
        float acc = 0.0f;
        const float* row = Wt + j;
        for (int i = 0; i < in; ++i) acc += x[i] * row[(size_t)i * out];
        y[j] = acc;
    }
}

/* y = Wt^T x. 输出列数大时按列分块并行, 每列求和顺序不变 => 逐位一致. */
static void gemv(float* y, const float* x, const float* Wt, int in, int out) {
    if (out >= 2048 && in >= 256) {
        const int nb = (out + 255) / 256;
        #pragma omp parallel for schedule(static)
        for (int b = 0; b < nb; ++b) {
            const int j0 = b * 256;
            const int jn = (j0 + 256 <= out) ? 256 : out - j0;
            gemv_blk(y, x, Wt, in, out, j0, jn);
        }
    } else {
        gemv_blk(y, x, Wt, in, out, 0, out);
    }
}

static void layernorm(float* y, const float* x, const float* g, const float* c,
                      int n, float eps) {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) sum += x[i];
    const float mean = sum / (float)n;
    float vsum = 0.0f;
    for (int i = 0; i < n; ++i) {
        const float d = x[i] - mean;
        vsum += d * d;
    }
    const float rstd = 1.0f / sqrtf(vsum / (float)n + eps);
    for (int i = 0; i < n; ++i) y[i] = (x[i] - mean) * rstd * g[i] + c[i];
}

/* S = S*w - (S kk) x (kk a) + v x k; y = S r. 逐 head 串行. */
static void head_step(int HS, int NH, float* S, const float* w2,
                      const float* k2, const float* v2, const float* a2,
                      const float* kk2, const float* r2, float* y2) {
    for (int h = 0; h < NH; ++h) {
        float* Sh = S + (size_t)h * HS * HS;
        const float* wh = w2 + (size_t)h * HS;
        const float* kh = k2 + (size_t)h * HS;
        const float* vh = v2 + (size_t)h * HS;
        const float* ah = a2 + (size_t)h * HS;
        const float* kkh = kk2 + (size_t)h * HS;
        const float* rh = r2 + (size_t)h * HS;
        float* yh = y2 + (size_t)h * HS;

        float buf[256];
        __m256 ss = _mm256_setzero_ps();
        for (int j = 0; j < HS; j += 8) {
            const __m256 t = _mm256_loadu_ps(kkh + j);
            ss = _mm256_fmadd_ps(t, t, ss);
        }
        float nrm = sqrtf(hsum256(ss));
        if (!(nrm > 1e-12f)) nrm = 1e-12f;
        const float inv = 1.0f / nrm;
        for (int j = 0; j < HS; ++j) buf[j] = kkh[j] * inv;

        for (int i = 0; i < HS; ++i) {
            float* row = Sh + (size_t)i * HS;
            __m256 proj = _mm256_setzero_ps();
            for (int j = 0; j < HS; j += 8)
                proj = _mm256_fmadd_ps(_mm256_loadu_ps(row + j),
                                       _mm256_loadu_ps(buf + j), proj);
            const __m256 pv = _mm256_set1_ps(hsum256(proj));
            const __m256 vi = _mm256_set1_ps(vh[i]);
            __m256 yacc = _mm256_setzero_ps();
            for (int j = 0; j < HS; j += 8) {
                const __m256 s8 = _mm256_loadu_ps(row + j);
                const __m256 kka = _mm256_mul_ps(_mm256_loadu_ps(buf + j),
                                                 _mm256_loadu_ps(ah + j));
                const __m256 vk = _mm256_mul_ps(vi, _mm256_loadu_ps(kh + j));
                const __m256 ns = _mm256_fmadd_ps(s8, _mm256_loadu_ps(wh + j),
                                                  _mm256_sub_ps(vk, _mm256_mul_ps(pv, kka)));
                _mm256_storeu_ps(row + j, ns);
                yacc = _mm256_fmadd_ps(ns, _mm256_loadu_ps(rh + j), yacc);
            }
            yh[i] = hsum256(yacc);
        }
    }
}

static void rk_layer(SnLayer* L, float* x, float* vfirst, float* dbg) {
    const int D = L->D, HS = L->HS, NH = L->NH, F = L->F;
    const int Rw = L->Rw, Ra = L->Ra, Rv = L->Rv, Rg = L->Rg;
    struct SnRWKV7* m = L->m;
    float* sc = m->sc;
    float* xn = sc;
    float* dd = xn + D;
    float *xr = dd + D, *xw = xr + D, *xk = xw + D, *xv = xk + D, *xa = xv + D, *xg = xa + D;
    float* r = xg + D;
    float* k = r + D;
    float* v = k + D;
    float* a = v + D;
    float* g = a + D;
    float* kk = g + D;
    float* ww = kk + D;
    float* y = ww + D;
    float* xn2 = y + D;
    float* xk2 = xn2 + D;
    float* lo = xk2 + D;      /* 256: LoRA 中间量 (最大 Rg) */
    float* ck = lo + 256;
    float* cksq = ck + F;

    layernorm(xn, x, L->ln1w, L->ln1b, D, m->eps);
    for (int i = 0; i < D; ++i) dd[i] = L->shift[i] - xn[i];
    {
        const float* lp = L->lerp;
        for (int i = 0; i < D; ++i) {
            const float d = dd[i];
            xr[i] = xn[i] + lp[i] * d;
            xw[i] = xn[i] + lp[D + i] * d;
            xk[i] = xn[i] + lp[2 * D + i] * d;
            xv[i] = xn[i] + lp[3 * D + i] * d;
            xa[i] = xn[i] + lp[4 * D + i] * d;
            xg[i] = xn[i] + lp[5 * D + i] * d;
        }
    }

    gemv(r, xr, L->rw, D, D);
    gemv(k, xk, L->kw, D, D);
    gemv(v, xv, L->vw, D, D);

    /* ww = exp(-sigmoid(tanh(xw W1) W2 + w0) / sqrt(e)) */
    gemv(lo, xw, L->w1, D, Rw);
    for (int i = 0; i < Rw; ++i) lo[i] = tanhf(lo[i]);
    gemv(dd, lo, L->w2, Rw, D);
    for (int i = 0; i < D; ++i)
        ww[i] = expf(-sgn(dd[i] + L->w0[i]) * SN_R2);

    /* v 混合: 首层写 v_first, 其余层用 sigmoid(xv V1 V2 + v0) 混合 */
    if (L->first) {
        memcpy(vfirst, v, sizeof(float) * D);
    } else {
        gemv(lo, xv, L->v1, D, Rv);
        gemv(dd, lo, L->v2, Rv, D);
        for (int i = 0; i < D; ++i) {
            const float mix = sgn(dd[i] + L->v0[i]);
            v[i] += (vfirst[i] - v[i]) * mix;
        }
    }

    /* a = sigmoid(xa A1 A2 + a0) */
    gemv(lo, xa, L->a1, D, Ra);
    gemv(dd, lo, L->a2, Ra, D);
    for (int i = 0; i < D; ++i) a[i] = sgn(dd[i] + L->a0[i]);

    /* g = sigmoid(xg G1) G2 */
    gemv(lo, xg, L->g1, D, Rg);
    for (int i = 0; i < Rg; ++i) lo[i] = sgn(lo[i]);
    gemv(g, lo, L->g2, Rg, D);

    for (int i = 0; i < D; ++i) {
        kk[i] = k[i] * L->kk_w[i];
        k[i] = k[i] + k[i] * (a[i] - 1.0f) * L->ka_w[i];
    }

    head_step(HS, NH, L->S, ww, k, v, a, kk, r, y);

    for (int h = 0; h < NH; ++h) {
        float* yh = y + (size_t)h * HS;
        const float* rh = r + (size_t)h * HS;
        const float* kh = k + (size_t)h * HS;
        const float* vh = v + (size_t)h * HS;
        const int base = h * HS;
        float mean = 0.0f;
        for (int i = 0; i < HS; ++i) mean += yh[i];
        mean /= (float)HS;
        float var = 0.0f;
        for (int i = 0; i < HS; ++i) {
            const float t = yh[i] - mean;
            var += t * t;
        }
        const float rstd = 1.0f / sqrtf(var / (float)HS + m->eps_group);
        float dot = 0.0f;
        for (int i = 0; i < HS; ++i) dot += rh[i] * kh[i] * L->rk_w[base + i];
        for (int i = 0; i < HS; ++i) {
            const float n = (yh[i] - mean) * rstd * L->lnx_w[base + i] + L->lnx_b[base + i];
            yh[i] = (n + dot * vh[i]) * g[base + i];
        }
    }

    gemv(dd, y, L->ow, D, D);
    for (int i = 0; i < D; ++i) x[i] += dd[i];
    memcpy(L->shift, xn, sizeof(float) * D);

    layernorm(xn2, x, L->ln2w, L->ln2b, D, m->eps);
    for (int i = 0; i < D; ++i)
        xk2[i] = xn2[i] + L->cmk_lerp[i] * (L->shift[D + i] - xn2[i]);
    gemv(ck, xk2, L->cmk, D, F);
    for (int i = 0; i < F; ++i) {
        const float t = ck[i] > 0.0f ? ck[i] : 0.0f;
        ck[i] = t;
        cksq[i] = t * t;
    }
    gemv(dd, cksq, L->cmv, F, D);
    for (int i = 0; i < D; ++i) x[i] += dd[i];
    memcpy(L->shift + D, xn2, sizeof(float) * D);

    if (dbg) {
        float* t = dbg;
        memcpy(t, xn, sizeof(float) * D); t += D;
        memcpy(t, r, sizeof(float) * D); t += D;
        memcpy(t, k, sizeof(float) * D); t += D;
        memcpy(t, v, sizeof(float) * D); t += D;
        memcpy(t, a, sizeof(float) * D); t += D;
        memcpy(t, g, sizeof(float) * D); t += D;
        memcpy(t, ww, sizeof(float) * D); t += D;
        memcpy(t, y, sizeof(float) * D); t += D;
        memcpy(t, x, sizeof(float) * D); t += D;
    }
}

int sn_rk_ready(const SnRWKV7* m) { return m && m->ready; }
int sn_rk_vocab(const SnRWKV7* m) { return m ? m->vocab : 0; }

void sn_rk_reset(SnRWKV7* m) {
    if (!m) return;
    for (int l = 0; l < m->L; ++l) {
        SnLayer* L = &m->layers[l];
        memset(L->S, 0, sizeof(float) * (size_t)L->NH * L->HS * L->HS);
        memset(L->shift, 0, sizeof(float) * 2 * L->D);
    }
}

void sn_rk_free(SnRWKV7* m) {
    if (!m) return;
    if (m->layers) {
        for (int l = 0; l < m->L; ++l) {
            free(m->layers[l].S);
            free(m->layers[l].shift);
        }
    }
    free(m->wbuf);
    free(m->layers);
    free(m->sc);
    free(m->top);
    free(m);
}

void sn_rk_step(SnRWKV7* m, int token, float* logits) {
    const int D = m->D;
    float* x = m->top;
    float* nrm = x + D;
    float* vfirst = nrm + D;
    memcpy(x, m->embed + (size_t)token * D, sizeof(float) * D);
    layernorm(x, x, m->en_w, m->en_b, D, m->eps);
    for (int l = 0; l < m->L; ++l)
        rk_layer(&m->layers[l], x, l == 0 ? vfirst : vfirst, NULL);
    layernorm(nrm, x, m->on_w, m->on_b, D, m->eps);
    gemv(logits, nrm, m->head_t, D, m->vocab);
}

void sn_rk_prefill(SnRWKV7* m, const int* tokens, int n, float* logits) {
    for (int t = 0; t < n; ++t) sn_rk_step(m, tokens[t], logits);
}

void sn_rk_dbg(SnRWKV7* m, int token, int layer, float* out) {
    const int D = m->D;
    float* x = m->top;
    float* vfirst = x + 2 * D;
    memcpy(x, m->embed + (size_t)token * D, sizeof(float) * D);
    layernorm(x, x, m->en_w, m->en_b, D, m->eps);
    for (int l = 0; l <= layer; ++l)
        rk_layer(&m->layers[l], x, vfirst, l == layer ? out : NULL);
}

static int rd_i32(FILE* f, int32_t* o, int n) {
    return (int)fread(o, sizeof(int32_t), (size_t)n, f) == n;
}

SnRWKV7* sn_rk_load(const char* path) {
    FILE* f = sn_fopen(path, "rb");
    if (!f) return NULL;
    int32_t h[10];
    if (!rd_i32(f, h, 10) || h[0] != SN_RK_MAGIC || h[1] != SN_RK_VERSION) {
        fclose(f);
        return NULL;
    }
    const int L = h[2], D = h[3], HS = h[4], NH = h[5], F = h[6], vocab = h[8];
    float hf[1];
    if (fread(hf, sizeof(float), 1, f) != 1) { fclose(f); return NULL; }

    SnRWKV7* m = (SnRWKV7*)calloc(1, sizeof(SnRWKV7));
    if (!m) { fclose(f); return NULL; }
    m->L = L; m->D = D; m->HS = HS; m->NH = NH; m->F = F; m->vocab = vocab;
    m->eps = hf[0];
    m->eps_group = SN_EPS_GROUP;

    int* ranks = (int*)malloc(sizeof(int) * (size_t)4 * L);
    m->layers = (SnLayer*)calloc((size_t)L, sizeof(SnLayer));
    if (!ranks || !m->layers) { free(ranks); sn_rk_free(m); fclose(f); return NULL; }
    if (!rd_i32(f, ranks, 4 * L)) { free(ranks); sn_rk_free(m); fclose(f); return NULL; }

    size_t per_layer = (size_t)19 * D + (size_t)4 * D * D + (size_t)2 * D * F;
    for (int l = 0; l < L; ++l)
        per_layer += (size_t)2 * D * (ranks[4 * l] + ranks[4 * l + 1] +
                                      ranks[4 * l + 2] + ranks[4 * l + 3]);
    const size_t total = (size_t)L * per_layer + 4 * (size_t)D +
                         (size_t)vocab * D + (size_t)D * vocab;
    m->wbuf = (float*)malloc(sizeof(float) * total);
    if (!m->wbuf) { free(ranks); sn_rk_free(m); fclose(f); return NULL; }

    float* p = m->wbuf;
    for (int l = 0; l < L; ++l) {
        SnLayer* A = &m->layers[l];
        A->m = m;
        A->D = D; A->HS = HS; A->NH = NH; A->F = F;
        A->first = (l == 0);
        A->Rw = ranks[4 * l]; A->Ra = ranks[4 * l + 1];
        A->Rv = ranks[4 * l + 2]; A->Rg = ranks[4 * l + 3];
        A->ln1w = p; p += D;
        A->ln1b = p; p += D;
        A->ln2w = p; p += D;
        A->ln2b = p; p += D;
        A->lerp = p; p += (size_t)6 * D;
        A->rw = p; p += (size_t)D * D;
        A->kw = p; p += (size_t)D * D;
        A->vw = p; p += (size_t)D * D;
        A->ow = p; p += (size_t)D * D;
        A->w1 = p; p += (size_t)D * A->Rw;
        A->w2 = p; p += (size_t)A->Rw * D;
        A->a1 = p; p += (size_t)D * A->Ra;
        A->a2 = p; p += (size_t)A->Ra * D;
        A->v1 = p; p += (size_t)D * A->Rv;
        A->v2 = p; p += (size_t)A->Rv * D;
        A->g1 = p; p += (size_t)D * A->Rg;
        A->g2 = p; p += (size_t)A->Rg * D;
        A->w0 = p; p += D;
        A->a0 = p; p += D;
        A->v0 = p; p += D;
        A->kk_w = p; p += D;
        A->ka_w = p; p += D;
        A->rk_w = p; p += D;
        A->lnx_w = p; p += D;
        A->lnx_b = p; p += D;
        A->cmk = p; p += (size_t)D * F;
        A->cmv = p; p += (size_t)F * D;
        A->cmk_lerp = p; p += D;
    }
    float* en_w = p; p += D;
    float* en_b = p; p += D;
    float* on_w = p; p += D;
    float* on_b = p; p += D;
    float* embed = p; p += (size_t)vocab * D;
    float* head_t = p; p += (size_t)D * vocab;

    if (fread(m->wbuf, sizeof(float), (size_t)(p - m->wbuf), f) != (size_t)(p - m->wbuf)) {
        free(ranks); sn_rk_free(m); fclose(f); return NULL;
    }
    fclose(f);
    free(ranks);

    m->en_w = en_w; m->en_b = en_b; m->on_w = on_w; m->on_b = on_b;
    m->embed = embed; m->head_t = head_t;

    for (int l = 0; l < L; ++l) {
        SnLayer* A = &m->layers[l];
        A->S = (float*)calloc((size_t)NH * HS * HS, sizeof(float));
        A->shift = (float*)calloc((size_t)2 * D, sizeof(float));
        if (!A->S || !A->shift) { sn_rk_free(m); return NULL; }
    }

    const size_t scn = (size_t)18 * D + 256 + (size_t)2 * F + 8;
    m->sc = (float*)calloc(scn, sizeof(float));
    m->top = (float*)calloc((size_t)3 * D, sizeof(float));
    if (!m->sc || !m->top) { sn_rk_free(m); return NULL; }
    m->ready = 1;
    return m;
}