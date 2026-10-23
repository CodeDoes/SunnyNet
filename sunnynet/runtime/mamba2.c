/* SunnyNet Runtime: Mamba-2 整模型单步解码 (AVX2/FMA). */
#include "mamba2.h"

#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#ifdef _WIN32
#include <windows.h>
static FILE* sn_fopen(const char* path, const char* mode) {
    wchar_t wp[4096], wm[8];
    if (!MultiByteToWideChar(CP_UTF8, 0, path, -1, wp, 4096)) return NULL;
    MultiByteToWideChar(CP_UTF8, 0, mode, -1, wm, 8);
    return _wfopen(wp, wm);
}
#else
static FILE* sn_fopen(const char* path, const char* mode) { return fopen(path, mode); }
#endif

void sn_m2_set_threads(int n) { if (n > 0) omp_set_num_threads(n); }
int sn_m2_get_threads(void) { return omp_get_max_threads(); }

#define SN_M2_MAGIC 0x324E4D53   /* "SMN2" */
#define SN_M2_VERSION 1

struct SnMamba2;

typedef struct SnLayer {
    int D, NH, HD, NS, NG, DI, CD, DT, conv;
    float eps;
    const float *nrm, *ip, *cw, *cb, *alog, *Dv, *dtb, *mn, *op;
    float* S;                    /* (NH, HD, NS) */
    float* ch;                   /* (conv-1, CD) 卷积历史 */
    struct SnMamba2* m;
} SnLayer;

struct SnMamba2 {
    int D, NH, HD, NS, NG, DI, CD, DT, L, vocab, conv, tied_head;
    float eps;
    const float* embed;
    const float* ln_f;
    const float* head;
    unsigned char* wbuf;
    SnLayer* layers;
    float* x;
    float* sc;
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

/* y = W x. W 存 (out,in) 行主序 (HF 原生布局), 按输出通道并行。 */
static void gemv(float* y, const float* x, const float* W, int in, int out) {
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < out; ++j) {
        const float* row = W + (size_t)j * in;
        __m256 acc = _mm256_setzero_ps();
        int i = 0;
        for (; i + 8 <= in; i += 8)
            acc = _mm256_fmadd_ps(_mm256_loadu_ps(row + i),
                                  _mm256_loadu_ps(x + i), acc);
        float s = hsum256(acc);
        for (; i < in; ++i) s += row[i] * x[i];
        y[j] = s;
    }
}

/* 无偏置 RMSNorm: y = x / rms(x) * g (均值平方)。 */
static void rms_norm(float* y, const float* x, const float* g, int n, float eps) {
    float ss = 0.0f;
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m256 v = _mm256_loadu_ps(x + i);
        ss += hsum256(_mm256_mul_ps(v, v));
    }
    for (; i < n; ++i) ss += x[i] * x[i];
    const __m256 r = _mm256_set1_ps(1.0f / sqrtf(ss / (float)n + eps));
    for (i = 0; i + 8 <= n; i += 8)
        _mm256_storeu_ps(y + i, _mm256_mul_ps(
            _mm256_mul_ps(_mm256_loadu_ps(x + i), r), _mm256_loadu_ps(g + i)));
    for (; i < n; ++i) y[i] = x[i] * _mm_cvtss_f32(_mm256_castps256_ps128(r)) * g[i];
}

static inline float silu(float x) { return x / (1.0f + expf(-x)); }

/* 数值稳定 softplus: log1p(exp(-|x|)) + max(x, 0)。dt 很大时 exp 会溢出。 */
static inline float softplus(float x) {
    const float a = fabsf(x);
    return log1pf(expf(-a)) + (x > 0.0f ? x : 0.0f);
}

/* 深度因果卷积 + silu. tmp 先拷原始输入: 历史要存卷积的输入而不是激活值。 */
static void dconv_silu(float* y, const float* x, float* tmp, float* hist,
                       const float* kw, const float* kb, int n, int conv) {
    memcpy(tmp, x, (size_t)n * sizeof(float));
    for (int c = 0; c < n; ++c) {
        float acc = kw[(size_t)c * conv + conv - 1] * tmp[c];
        for (int t = 0; t < conv - 1; ++t)
            acc += kw[(size_t)c * conv + t] * hist[(size_t)t * n + c];
        y[c] = silu(acc + kb[c]);
    }
    if (conv > 1) {
        if (conv > 2) memmove(hist, hist + n, (size_t)(conv - 2) * n * sizeof(float));
        memcpy(hist + (size_t)(conv - 2) * n, tmp, (size_t)n * sizeof(float));
    }
}

/* 状态更新 + 读出. 逐 (head, head_dim) 行独立: 行内存 (NS,) 连续。
 * dA 与 dt 沿 head_dim 广播, 故每行一个标量系数, 无需矩阵乘。
 * B/C 按 group 共享 (n_groups 可 < n_heads), 同一 group 的 head 读同一段。 */
static void m2_scan(SnLayer* A, const float* xssm, const float* B, const float* C,
                    const float* dA, const float* dth, float* y) {
    const int NH = A->NH, HD = A->HD, NS = A->NS;
    const int HG = NH / A->NG;
    #pragma omp parallel for schedule(static)
    for (int hp = 0; hp < NH * HD; ++hp) {
        const int h = hp / HD;
        const float* Bg = B + (size_t)(h / HG) * NS;
        const float* Cg = C + (size_t)(h / HG) * NS;
        float* row = A->S + (size_t)hp * NS;
        const float da = dA[hp];
        const float co = dth[h] * xssm[hp];
        int n = 0;
        for (; n + 8 <= NS; n += 8) {
            const __m256 s = _mm256_fmadd_ps(_mm256_set1_ps(co),
                                             _mm256_loadu_ps(Bg + n),
                                             _mm256_mul_ps(_mm256_loadu_ps(row + n),
                                                           _mm256_set1_ps(da)));
            _mm256_storeu_ps(row + n, s);
        }
        for (; n < NS; ++n) row[n] = row[n] * da + co * Bg[n];
        float acc = 0.0f;
        n = 0;
        for (; n + 8 <= NS; n += 8)
            acc += hsum256(_mm256_mul_ps(_mm256_loadu_ps(row + n),
                                         _mm256_loadu_ps(Cg + n)));
        for (; n < NS; ++n) acc += row[n] * Cg[n];
        y[hp] = acc;
    }
}

/* ---------------------------------------------------------------- 加载 */

typedef struct { const unsigned char* p; } Cur;

static const float* take_f32(Cur* c, size_t n) {
    const float* r = (const float*)c->p;
    c->p += n * sizeof(float);
    return r;
}

static int rd_i32(FILE* f, int32_t* o, int n) {
    return (int)fread(o, sizeof(int32_t), (size_t)n, f) == n;
}

SnMamba2* sn_m2_load(const char* path) {
    FILE* f = sn_fopen(path, "rb");
    if (!f) return NULL;
    int32_t h[8];
    if (!rd_i32(f, h, 8) || h[0] != SN_M2_MAGIC || h[1] != SN_M2_VERSION) {
        fclose(f);
        return NULL;
    }
    const int L = h[2], D = h[3], NH = h[4], HD = h[5], NS = h[6], NG = h[7];
    float hf[1];
    if (fread(hf, sizeof(float), 1, f) != 1) { fclose(f); return NULL; }
    int32_t meta[3];
    if (!rd_i32(f, meta, 3)) { fclose(f); return NULL; }   /* vocab, conv, tied */

    const long pos = ftell(f);
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    const size_t nbytes = (size_t)(ftell(f) - pos);
    unsigned char* buf = (unsigned char*)malloc(nbytes);
    if (!buf) { fclose(f); return NULL; }
    fseek(f, pos, SEEK_SET);
    if (fread(buf, 1, nbytes, f) != nbytes) { free(buf); fclose(f); return NULL; }
    fclose(f);

    SnMamba2* m = (SnMamba2*)calloc(1, sizeof(SnMamba2));
    if (!m) { free(buf); return NULL; }
    m->layers = (SnLayer*)calloc((size_t)L, sizeof(SnLayer));
    if (!m->layers) { free(buf); free(m); return NULL; }
    m->L = L; m->D = D; m->NH = NH; m->HD = HD; m->NS = NS; m->NG = NG;
    m->DI = NH * HD;
    m->CD = m->DI + 2 * NG * NS;
    m->DT = NH;
    m->vocab = meta[0]; m->conv = meta[1]; m->tied_head = meta[2];
    m->eps = hf[0];

    Cur c = { buf };
    for (int l = 0; l < L; ++l) {
        SnLayer* A = &m->layers[l];
        A->m = m;
        A->D = D; A->NH = NH; A->HD = HD; A->NS = NS; A->NG = NG;
        A->DI = m->DI; A->CD = m->CD; A->DT = m->DT; A->conv = m->conv;
        A->eps = m->eps;
        /* 顺序由导出器 _export_mamba2.py 决定, 两处必须同步改 */
        A->nrm = take_f32(&c, D);
        A->ip = take_f32(&c, (size_t)(m->DI + m->CD + m->DT) * D);
        A->cw = take_f32(&c, (size_t)m->CD * m->conv);
        A->cb = take_f32(&c, m->CD);
        A->alog = take_f32(&c, NH);
        A->Dv = take_f32(&c, NH);
        A->dtb = take_f32(&c, NH);
        A->mn = take_f32(&c, m->DI);
        A->op = take_f32(&c, (size_t)D * m->DI);
    }
    m->embed = take_f32(&c, (size_t)m->vocab * D);
    m->ln_f = take_f32(&c, D);
    m->head = m->tied_head ? NULL : take_f32(&c, (size_t)m->vocab * D);

    for (int l = 0; l < L; ++l) {
        SnLayer* A = &m->layers[l];
        A->S = (float*)calloc((size_t)NH * HD * NS, sizeof(float));
        A->ch = (float*)calloc((size_t)(m->conv - 1) * m->CD, sizeof(float));
        if (!A->S || !A->ch) { sn_m2_free(m); return NULL; }
    }
    /* 暂存: h(D) proj(DI+CD+DT) act(CD) tmp(CD) dA(DI) dth(NH) y(DI) o(D) */
    const size_t scn = (size_t)D + (size_t)(m->DI + m->CD + m->DT)
                       + (size_t)m->CD * 2 + (size_t)m->DI * 2 + (size_t)m->DT
                       + (size_t)D + 16;
    m->x = (float*)calloc((size_t)D, sizeof(float));
    m->sc = (float*)calloc(scn, sizeof(float));
    if (!m->x || !m->sc) { sn_m2_free(m); return NULL; }
    m->wbuf = buf;
    m->ready = 1;
    return m;
}

void sn_m2_free(SnMamba2* m) {
    if (!m) return;
    if (m->layers) {
        for (int l = 0; l < m->L; ++l) {
            free(m->layers[l].S);
            free(m->layers[l].ch);
        }
        free(m->layers);
    }
    free(m->x);
    free(m->sc);
    free(m->wbuf);
    free(m);
}

void sn_m2_reset(SnMamba2* m) {
    if (!m) return;
    for (int l = 0; l < m->L; ++l) {
        SnLayer* A = &m->layers[l];
        memset(A->S, 0, (size_t)m->NH * m->HD * m->NS * sizeof(float));
        memset(A->ch, 0, (size_t)(m->conv - 1) * m->CD * sizeof(float));
    }
}

int sn_m2_ready(const SnMamba2* m) { return m && m->ready; }
int sn_m2_vocab(const SnMamba2* m) { return m ? m->vocab : 0; }

static void m2_layer(SnLayer* A, float* x) {
    const int D = A->D, NH = A->NH, HD = A->HD, NS = A->NS, NG = A->NG;
    const int DI = A->DI, CD = A->CD, DT = A->DT;
    float* sc = A->m->sc;
    float* h = sc;
    float* proj = h + D;
    float* act = proj + DI + CD + DT;
    float* tmp = act + CD;
    float* dA = tmp + CD;
    float* dth = dA + DI;
    float* y = dth + NH;
    float* o = y + DI;

    rms_norm(h, x, A->nrm, D, A->eps);
    gemv(proj, h, A->ip, D, DI + CD + DT);
    const float* gate = proj;
    const float* bc = proj + DI;
    const float* dt = proj + DI + CD;
    dconv_silu(act, bc, tmp, A->ch, A->cw, A->cb, CD, A->conv);

    float* xssm = act;
    const float* B = act + DI;
    const float* C = act + DI + NG * NS;

    /* dA[h][p] = exp(dt_h * A_h), 沿 head_dim 全同 */
    for (int hh = 0; hh < NH; ++hh) {
        const float d = softplus(dt[hh] + A->dtb[hh]);
        const float a = -expf(A->alog[hh]);
        const float v = expf(d * a);
        for (int p = 0; p < HD; ++p) dA[hh * HD + p] = v;
        dth[hh] = d;
    }
    m2_scan(A, xssm, B, C, dA, dth, y);
    for (int hh = 0; hh < NH; ++hh) {
        const float d = A->Dv[hh];
        for (int p = 0; p < HD; ++p) y[hh * HD + p] += d * xssm[hh * HD + p];
    }
    for (int i = 0; i < DI; ++i) y[i] *= silu(gate[i]);
    rms_norm(y, y, A->mn, DI, A->eps);
    gemv(o, y, A->op, DI, D);
    for (int i = 0; i < D; ++i) x[i] += o[i];
}

static void m2_forward(SnMamba2* m, int token, float* logits) {
    const int D = m->D;
    float* x = m->x;
    memcpy(x, m->embed + (size_t)token * D, (size_t)D * sizeof(float));
    for (int l = 0; l < m->L; ++l) m2_layer(&m->layers[l], x);
    rms_norm(x, x, m->ln_f, D, m->eps);
    gemv(logits, x, m->head ? m->head : m->embed, D, m->vocab);
}

void sn_m2_step(SnMamba2* m, int token, float* logits) {
    if (!m || !m->ready) return;
    m2_forward(m, token, logits);
}

void sn_m2_prefill(SnMamba2* m, const int* tokens, int n, float* logits) {
    for (int i = 0; i < n; ++i) sn_m2_step(m, tokens[i], logits);
}

/* 跑完第 layer 层后把主流 x 拷出; layer<0 表示跑完全部层。 */
void sn_m2_dbg(SnMamba2* m, int token, int layer, float* out) {
    if (!m || !m->ready) return;
    const int D = m->D;
    memcpy(m->x, m->embed + (size_t)token * D, (size_t)D * sizeof(float));
    const int n = layer < 0 ? m->L : layer + 1;
    for (int l = 0; l < n && l < m->L; ++l) m2_layer(&m->layers[l], m->x);
    memcpy(out, m->x, (size_t)D * sizeof(float));
}