/* SunnyNet Runtime: RWKV-7 单步解码, 整模型在 C (AVX2/FMA).
 *
 * 权重矩阵按 gemv 约定存 (in,out) 行主序: y = Wt^T x; 或 INT4 量化存 (out,in).
 * 状态 S 逐 head (HS,HS); rank-1 校正项与衰减项共用更新前的 S.
 * w 的 LoRA 中间夹 tanh, 不可折叠; a/v 无夹层但保留分解.
 * 单线程: B=1 单序列, 张量太小, 起线程的唤醒开销大于收益.
 */
#include "rwkv7.h"
#include "quant.h"

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
#define SN_RK_VERSION 3
#define SN_R2 0.6065306597126334f /* 1/sqrt(e) */
#define SN_EPS_GROUP 6.4e-4f
#define SN_Q4_BLOCK 64

/* fp32 (in,out) 或 INT4 (out,in) 的权重矩阵. */
typedef struct {
    int is_q4, in, out;
    const float* w;             /* fp32: (in,out) */
    const unsigned char* q;     /* INT4: (out, in/2) */
    const float* scale;         /* INT4: (out, in/64) */
} Wmat;

struct SnRWKV7;

typedef struct {
    int D, HS, NH, F, Rw, Ra, Rv, Rg;
    const float *ln1w, *ln1b, *ln2w, *ln2b;
    const float* lerp;              /* (6,D) */
    Wmat rw, kw, vw, ow;            /* (D,D) */
    Wmat w1, w2, a1, a2, v1, v2, g1, g2;
    Wmat cmk, cmv;
    const float *w0, *a0, *v0;
    const float *kk_w, *ka_w, *rk_w;
    const float *lnx_w, *lnx_b;
    const float* cmk_lerp;
    float* S;                       /* (NH,HS,HS) */
    float* shift;                   /* (2,D) */
    int first;
    struct SnRWKV7* m;
} SnLayer;

struct SnRWKV7 {
    int vocab, D, HS, NH, F, L, is_q4;
    float eps, eps_group;
    unsigned char* wbuf;
    SnLayer* layers;
    const float *en_w, *en_b, *on_w, *on_b;
    const float* embed;   /* (vocab,D) */
    Wmat head;            /* (D,vocab) */
    float* sc;
    float* top;
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

/* y = Wt^T x, Wt 存 (in,out). */
static void gemv(float* y, const float* x, const float* Wt, int in, int out) {
    int j = 0;
    for (; j + 32 <= out; j += 32) {
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
    for (; j + 8 <= out; j += 8) {
        __m256 acc = _mm256_setzero_ps();
        for (int i = 0; i < in; ++i) {
            const __m256 b = _mm256_set1_ps(x[i]);
            acc = _mm256_fmadd_ps(b, _mm256_loadu_ps(Wt + (size_t)i * out + j), acc);
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

static void wmat_mul(float* y, const float* x, const Wmat* M) {
    if (M->is_q4) sn_gemv_i4(y, x, M->q, M->scale, M->in, M->out, SN_Q4_BLOCK);
    else gemv(y, x, M->w, M->in, M->out);
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
    float* lo = xk2 + D;      /* 256: LoRA 中间量 */
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

    wmat_mul(r, xr, &L->rw);
    wmat_mul(k, xk, &L->kw);
    wmat_mul(v, xv, &L->vw);

    /* ww = exp(-sigmoid(tanh(xw W1) W2 + w0) / sqrt(e)) */
    wmat_mul(lo, xw, &L->w1);
    for (int i = 0; i < Rw; ++i) lo[i] = tanhf(lo[i]);
    wmat_mul(dd, lo, &L->w2);
    for (int i = 0; i < D; ++i)
        ww[i] = expf(-sgn(dd[i] + L->w0[i]) * SN_R2);

    /* v 混合: 首层写 v_first, 其余层用 sigmoid(xv V1 V2 + v0) 混合 */
    if (L->first) {
        memcpy(vfirst, v, sizeof(float) * D);
    } else {
        wmat_mul(lo, xv, &L->v1);
        wmat_mul(dd, lo, &L->v2);
        for (int i = 0; i < D; ++i) {
            const float mix = sgn(dd[i] + L->v0[i]);
            v[i] += (vfirst[i] - v[i]) * mix;
        }
    }

    /* a = sigmoid(xa A1 A2 + a0) */
    wmat_mul(lo, xa, &L->a1);
    wmat_mul(dd, lo, &L->a2);
    for (int i = 0; i < D; ++i) a[i] = sgn(dd[i] + L->a0[i]);

    /* g = sigmoid(xg G1) G2 */
    wmat_mul(lo, xg, &L->g1);
    for (int i = 0; i < Rg; ++i) lo[i] = sgn(lo[i]);
    wmat_mul(g, lo, &L->g2);

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

    wmat_mul(dd, y, &L->ow);
    for (int i = 0; i < D; ++i) x[i] += dd[i];
    memcpy(L->shift, xn, sizeof(float) * D);

    layernorm(xn2, x, L->ln2w, L->ln2b, D, m->eps);
    for (int i = 0; i < D; ++i)
        xk2[i] = xn2[i] + L->cmk_lerp[i] * (L->shift[D + i] - xn2[i]);
    wmat_mul(ck, xk2, &L->cmk);
    for (int i = 0; i < F; ++i) {
        const float t = ck[i] > 0.0f ? ck[i] : 0.0f;
        ck[i] = t;
        cksq[i] = t * t;
    }
    wmat_mul(dd, cksq, &L->cmv);
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
        rk_layer(&m->layers[l], x, vfirst, NULL);
    layernorm(nrm, x, m->on_w, m->on_b, D, m->eps);
    wmat_mul(logits, nrm, &m->head);
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

/* ---------------------------------------------------------------- 加载 */

typedef struct { const unsigned char* p; } Cur;

static void take_wmat(Cur* c, Wmat* M, int in, int out) {
    int tag;
    memcpy(&tag, c->p, sizeof(int));
    c->p += sizeof(int);
    M->in = in;
    M->out = out;
    M->w = NULL;
    M->q = NULL;
    M->scale = NULL;
    if (tag == 1) {
        const size_t qb = (size_t)out * (size_t)(in / 2);
        const size_t sb = (size_t)out * (size_t)(in / SN_Q4_BLOCK) * sizeof(float);
        M->is_q4 = 1;
        M->q = c->p;
        c->p += qb;
        M->scale = (const float*)c->p;
        c->p += sb;
    } else {
        M->is_q4 = 0;
        M->w = (const float*)c->p;
        c->p += (size_t)in * out * sizeof(float);
    }
}

static const float* take_f32(Cur* c, size_t n) {
    c->p += sizeof(int);   /* 标签, f32 恒为 0 */
    const float* r = (const float*)c->p;
    c->p += n * sizeof(float);
    return r;
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
    const int file_q4 = h[9];
    float hf[1];
    if (fread(hf, sizeof(float), 1, f) != 1) { fclose(f); return NULL; }

    int* ranks = (int*)malloc(sizeof(int) * (size_t)4 * L);
    if (!ranks || !rd_i32(f, ranks, 4 * L)) { free(ranks); fclose(f); return NULL; }

    /* 余下全部载荷一次读入, 之后在内存里按标志游标解析. */
    const long pos = ftell(f);
    if (fseek(f, 0, SEEK_END) != 0) { free(ranks); fclose(f); return NULL; }
    const long end = ftell(f);
    const size_t nbytes = (size_t)(end - pos);
    unsigned char* buf = (unsigned char*)malloc(nbytes);
    if (!buf) { free(ranks); fclose(f); return NULL; }
    fseek(f, pos, SEEK_SET);
    if (fread(buf, 1, nbytes, f) != nbytes) { free(buf); free(ranks); fclose(f); return NULL; }
    fclose(f);

    SnRWKV7* m = (SnRWKV7*)calloc(1, sizeof(SnRWKV7));
    m->layers = (SnLayer*)calloc((size_t)L, sizeof(SnLayer));
    if (!m || !m->layers) { free(buf); free(ranks); free(m); return NULL; }
    m->L = L; m->D = D; m->HS = HS; m->NH = NH; m->F = F; m->vocab = vocab;
    m->eps = hf[0];
    m->eps_group = SN_EPS_GROUP;
    m->is_q4 = file_q4;
    m->wbuf = buf;

    Cur c = { buf };
    for (int l = 0; l < L; ++l) {
        SnLayer* A = &m->layers[l];
        A->m = m;
        A->D = D; A->HS = HS; A->NH = NH; A->F = F; A->first = (l == 0);
        A->Rw = ranks[4 * l]; A->Ra = ranks[4 * l + 1];
        A->Rv = ranks[4 * l + 2]; A->Rg = ranks[4 * l + 3];
        A->ln1w = take_f32(&c, D);
        A->ln1b = take_f32(&c, D);
        A->ln2w = take_f32(&c, D);
        A->ln2b = take_f32(&c, D);
        A->lerp = take_f32(&c, (size_t)6 * D);
        take_wmat(&c, &A->rw, D, D);
        take_wmat(&c, &A->kw, D, D);
        take_wmat(&c, &A->vw, D, D);
        take_wmat(&c, &A->ow, D, D);
        take_wmat(&c, &A->w1, D, A->Rw);
        take_wmat(&c, &A->w2, A->Rw, D);
        take_wmat(&c, &A->a1, D, A->Ra);
        take_wmat(&c, &A->a2, A->Ra, D);
        take_wmat(&c, &A->v1, D, A->Rv);
        take_wmat(&c, &A->v2, A->Rv, D);
        take_wmat(&c, &A->g1, D, A->Rg);
        take_wmat(&c, &A->g2, A->Rg, D);
        A->w0 = take_f32(&c, D);
        A->a0 = take_f32(&c, D);
        A->v0 = take_f32(&c, D);
        A->kk_w = take_f32(&c, D);
        A->ka_w = take_f32(&c, D);
        A->rk_w = take_f32(&c, D);
        A->lnx_w = take_f32(&c, D);
        A->lnx_b = take_f32(&c, D);
        take_wmat(&c, &A->cmk, D, F);
        take_wmat(&c, &A->cmv, F, D);
        A->cmk_lerp = take_f32(&c, D);
    }
    m->en_w = take_f32(&c, D);
    m->en_b = take_f32(&c, D);
    m->on_w = take_f32(&c, D);
    m->on_b = take_f32(&c, D);
    m->embed = take_f32(&c, (size_t)vocab * D);
    take_wmat(&c, &m->head, D, vocab);
    free(ranks);

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