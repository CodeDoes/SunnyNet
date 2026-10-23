/* SunnyNet Runtime: DeltaNet 整模型单步解码 (AVX2/FMA). */
#include "deltanet.h"
#include "quant.h"

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

void sn_dn_set_threads(int n) { if (n > 0) omp_set_num_threads(n); }
int sn_dn_get_threads(void) { return omp_get_max_threads(); }

#define SN_DN_MAGIC 0x4E4E4444   /* "DNND" */
#define SN_DN_MAGIC_Q4 0x51444E44 /* "QNDD" 带 INT4 标签 */
#define SN_DN_VERSION 1
#define SN_DN_BLOCK SN_I4_BLOCK

struct SnLayer;

/* GEMV 矩阵. fp32 时 w 有效; INT4 时 q/scale 有效 (in 按 block 分组). */
typedef struct {
    const float* w;
    const unsigned char* q;
    const float* scale;
    int is_q4, in, out;
} DW;

struct SnDeltaNet {
    int D, NH, HK, HV, F, L, vocab, conv, tied_head;
    float eps;
    const float* embed;      /* (vocab, D) */
    const float* ln_f;       /* (D,) */
    DW head;                 /* (vocab, D); tied 时全空 */
    unsigned char* wbuf;
    struct SnLayer* layers;
    float* x;                /* 残差主流 (D,) */
    float* sc;               /* 层内暂存 */
    int ready;
};

typedef struct SnLayer {
    int D, NH, HK, HV, KV, VV, F, conv;
    float eps;
    const float *an, *mn;             /* attn_norm / mlp_norm (D,) */
    DW qp, kp, vp, op;                /* 投影 (out,in), y = W x */
    const float *qc, *kc, *vc;        /* 短卷积核 (n, conv), 不量化 */
    const float *on;                  /* o_norm (HV,) */
    const float *bp;                  /* b_proj (NH, D), out=NH 太小不量化 */
    DW gp, up, dp;                    /* MLP: (F,D) (F,D) (D,F) */
    float* S;                         /* (NH, HK, HV) */
    float* cq, *ck, *cv;              /* (conv-1, KV/VV) 历史 */
    SnDeltaNet* m;
} SnLayer;

static inline float hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    return _mm_cvtss_f32(lo);
}

/* y = W x. W 存 (out,in) 行主序 (HF 原生布局), 按输出通道并行。
 * 行内连续读, 与 sunnynet.c 的做法一致; 每行一个输出, 无跨线程依赖。 */
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

/* 按 DW 的存储标志分发: INT4 走 quant.c 的 maddubs 路径。 */
static inline void gemv_w(float* y, const float* x, const DW* W) {
    if (W->is_q4)
        sn_gemv_i4(y, x, W->q, W->scale, W->in, W->out, SN_DN_BLOCK);
    else
        gemv(y, x, W->w, W->in, W->out);
}

/* 无偏置 RMSNorm: y = x / rms(x) * g (均值平方, 同 HF LlamaRMSNorm)。 */
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

/* 深度因果卷积 + silu; hist 为 (conv-1, n) 原地滑动。
 * tmp 由调用方给, 因为 x 与 y 常是同一块缓冲: 历史要存卷积的**输入**,
 * 原地写回后 x 已是 silu 值, 直接 memcpy 会把错的值存进历史。 */
static void dconv_silu(float* y, const float* x, float* tmp, float* hist,
                       const float* kw, int n, int conv) {
    memcpy(tmp, x, (size_t)n * sizeof(float));
    for (int c = 0; c < n; ++c) {
        float acc = kw[(size_t)c * conv + conv - 1] * tmp[c];
        for (int t = 0; t < conv - 1; ++t)
            acc += kw[(size_t)c * conv + t] * hist[(size_t)t * n + c];
        y[c] = silu(acc);
    }
    if (conv > 1) {
        if (conv > 2) memmove(hist, hist + n, (size_t)(conv - 2) * n * sizeof(float));
        memcpy(hist + (size_t)(conv - 2) * n, tmp, (size_t)n * sizeof(float));
    }
}

/* 整段 L2 归一化, 再乘 scale。范数必须覆盖整个 head, 不能按向量宽度分块各算各的。 */
static void l2norm(float* x, int n, float scale) {
    float ss = 0.0f;
    int h = 0;
    for (; h + 8 <= n; h += 8) {
        const __m256 v = _mm256_loadu_ps(x + h);
        ss += hsum256(_mm256_mul_ps(v, v));
    }
    for (; h < n; ++h) ss += x[h] * x[h];
    const __m256 r = _mm256_set1_ps(scale / sqrtf(ss + 1e-6f));
    for (h = 0; h + 8 <= n; h += 8)
        _mm256_storeu_ps(x + h, _mm256_mul_ps(_mm256_loadu_ps(x + h), r));
    const float rs = _mm_cvtss_f32(_mm256_castps256_ps128(r));
    for (; h < n; ++h) x[h] *= rs;
}

/* delta rule 单步 + 读出. S 按 (HK,HV) 行主序, 对应标准写法的 S^T。
 * kv[j] = sum_i S[i][j] k[i]; d[j] = beta*(v[j]-kv[j]);
 * S[i][j] += k[i] d[j]; o[j] = sum_i S[i][j] q[i] */
static void dn_attn(SnLayer* A, float* q, float* k, float* v,
                    float* beta, float* o, float* kv, float* d) {
    const int NH = A->NH, HK = A->HK, HV = A->HV;
    #pragma omp parallel for schedule(static)
    for (int h = 0; h < NH; ++h) {
        float* S = A->S + (size_t)h * HK * HV;
        const float* kh = k + (size_t)h * HK;
        const float* vh = v + (size_t)h * HV;
        const float* qh = q + (size_t)h * HK;
        float* oh = o + (size_t)h * HV;
        float* kvh = kv + (size_t)h * HV;    /* 每 head 一段, 免跨线程共享 */
        float* dh = d + (size_t)h * HV;
        const float b = beta[h];

        memset(kvh, 0, (size_t)HV * sizeof(float));
        for (int i = 0; i < HK; ++i) {
            const __m256 ki = _mm256_set1_ps(kh[i]);
            const float* row = S + (size_t)i * HV;
            int j = 0;
            for (; j + 8 <= HV; j += 8)
                _mm256_storeu_ps(kvh + j, _mm256_fmadd_ps(
                    ki, _mm256_loadu_ps(row + j), _mm256_loadu_ps(kvh + j)));
            for (; j < HV; ++j) kvh[j] += kh[i] * row[j];
        }
        for (int j = 0; j < HV; ++j) dh[j] = b * (vh[j] - kvh[j]);

        memset(oh, 0, (size_t)HV * sizeof(float));
        for (int i = 0; i < HK; ++i) {
            const __m256 ki = _mm256_set1_ps(kh[i]);
            const __m256 qi = _mm256_set1_ps(qh[i]);
            float* row = S + (size_t)i * HV;
            int j = 0;
            for (; j + 8 <= HV; j += 8) {
                const __m256 s = _mm256_fmadd_ps(ki, _mm256_loadu_ps(dh + j),
                                                 _mm256_loadu_ps(row + j));
                _mm256_storeu_ps(row + j, s);
                _mm256_storeu_ps(oh + j, _mm256_fmadd_ps(
                    s, qi, _mm256_loadu_ps(oh + j)));
            }
            for (; j < HV; ++j) {
                const float s = row[j] + kh[i] * dh[j];
                row[j] = s;
                oh[j] += s * qh[i];
            }
        }
    }
}

/* ---------------------------------------------------------------- 加载 */

typedef struct { const unsigned char* p; } Cur;

static const float* take_f32(Cur* c, size_t n) {
    const float* r = (const float*)c->p;
    c->p += n * sizeof(float);
    return r;
}

/* 带标签的 GEMV 矩阵. 标签 1: INT4 = q(out,in/2 字节) + scale(out,in/block);
 * 标签 0: fp32 (out,in). 仅 Q4 magic 的文件有标签, fp32 文件仍是无标签的裸流。 */
static void take_mat(Cur* c, DW* W, int in, int out, int tagged) {
    W->in = in;
    W->out = out;
    W->w = NULL;
    W->q = NULL;
    W->scale = NULL;
    W->is_q4 = 0;
    int tag = 0;
    if (tagged) {
        memcpy(&tag, c->p, sizeof(int));
        c->p += sizeof(int);
    }
    if (tag == 1) {
        const size_t qb = (size_t)out * (size_t)(in / 2);
        const size_t sb = (size_t)out * (size_t)(in / SN_DN_BLOCK) * sizeof(float);
        W->is_q4 = 1;
        W->q = c->p;
        c->p += qb;
        W->scale = (const float*)c->p;
        c->p += sb;
    } else {
        W->w = (const float*)c->p;
        c->p += (size_t)in * out * sizeof(float);
    }
}

static int rd_i32(FILE* f, int32_t* o, int n) {
    return (int)fread(o, sizeof(int32_t), (size_t)n, f) == n;
}

SnDeltaNet* sn_dn_load(const char* path) {
    FILE* f = sn_fopen(path, "rb");
    if (!f) return NULL;
    int32_t h[8];
    if (!rd_i32(f, h, 8)
        || (h[0] != SN_DN_MAGIC && h[0] != SN_DN_MAGIC_Q4)
        || h[1] != SN_DN_VERSION) {
        fclose(f);
        return NULL;
    }
    const int tagged = (h[0] == SN_DN_MAGIC_Q4);
    const int L = h[2], D = h[3], NH = h[4], HK = h[5], HV = h[6], F = h[7];
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

    SnDeltaNet* m = (SnDeltaNet*)calloc(1, sizeof(SnDeltaNet));
    if (!m) { free(buf); return NULL; }
    m->layers = (SnLayer*)calloc((size_t)L, sizeof(SnLayer));
    if (!m->layers) { free(buf); free(m); return NULL; }
    m->L = L; m->D = D; m->NH = NH; m->HK = HK; m->HV = HV; m->F = F;
    m->vocab = meta[0]; m->conv = meta[1]; m->tied_head = meta[2];
    m->eps = hf[0];

    Cur c = { buf };
    for (int l = 0; l < L; ++l) {
        SnLayer* A = &m->layers[l];
        A->m = m;
        A->D = D; A->NH = NH; A->HK = HK; A->HV = HV;
        A->KV = NH * HK; A->VV = NH * HV; A->F = F; A->conv = m->conv;
        A->eps = m->eps;
        /* 顺序由导出器 _export_deltanet.py 决定, 两处必须同步改 */
        A->an = take_f32(&c, D);
        take_mat(&c, &A->qp, D, D, tagged);
        take_mat(&c, &A->kp, D, D, tagged);
        take_mat(&c, &A->vp, D, D, tagged);
        take_mat(&c, &A->op, D, D, tagged);
        A->qc = take_f32(&c, (size_t)A->KV * m->conv);
        A->kc = take_f32(&c, (size_t)A->KV * m->conv);
        A->vc = take_f32(&c, (size_t)A->VV * m->conv);
        A->on = take_f32(&c, HV);
        A->bp = take_f32(&c, (size_t)NH * D);
        A->mn = take_f32(&c, D);
        take_mat(&c, &A->gp, D, F, tagged);
        take_mat(&c, &A->up, D, F, tagged);
        take_mat(&c, &A->dp, F, D, tagged);
    }
    m->embed = take_f32(&c, (size_t)m->vocab * D);
    m->ln_f = take_f32(&c, D);
    if (!m->tied_head)
        take_mat(&c, &m->head, D, m->vocab, tagged);
    else {
        m->head.w = NULL; m->head.q = NULL; m->head.scale = NULL;
        m->head.is_q4 = 0; m->head.in = D; m->head.out = m->vocab;
    }

    for (int l = 0; l < L; ++l) {
        SnLayer* A = &m->layers[l];
        A->S = (float*)calloc((size_t)NH * HK * HV, sizeof(float));
        A->cq = (float*)calloc((size_t)(m->conv - 1) * A->KV, sizeof(float));
        A->ck = (float*)calloc((size_t)(m->conv - 1) * A->KV, sizeof(float));
        A->cv = (float*)calloc((size_t)(m->conv - 1) * A->VV, sizeof(float));
        if (!A->S || !A->cq || !A->ck || !A->cv) { sn_dn_free(m); return NULL; }
    }
    /* 暂存: h(D) q(KV) k(KV) v(VV) o(VV) beta(NH) kv(NH*HV) d(NH*HV) g(F) u(F) */
    const size_t scn = (size_t)D + 2 * (size_t)(NH * HK) + 2 * (size_t)(NH * HV)
                       + NH + 2 * (size_t)(NH * HV) + 2 * F + 16;
    m->x = (float*)calloc((size_t)D, sizeof(float));
    m->sc = (float*)calloc(scn, sizeof(float));
    if (!m->x || !m->sc) { sn_dn_free(m); return NULL; }
    /* head 与 embed 都是 (vocab,D), 正是 gemv 要的 (out,in) 布局, 无需转置 */
    m->wbuf = buf;
    m->ready = 1;
    return m;
}

void sn_dn_free(SnDeltaNet* m) {
    if (!m) return;
    if (m->layers) {
        for (int l = 0; l < m->L; ++l) {
            free(m->layers[l].S);
            free(m->layers[l].cq);
            free(m->layers[l].ck);
            free(m->layers[l].cv);
        }
        free(m->layers);
    }
    free(m->x);
    free(m->sc);
    free(m->wbuf);
    free(m);
}

void sn_dn_reset(SnDeltaNet* m) {
    if (!m) return;
    for (int l = 0; l < m->L; ++l) {
        SnLayer* A = &m->layers[l];
        memset(A->S, 0, (size_t)m->NH * m->HK * m->HV * sizeof(float));
        memset(A->cq, 0, (size_t)(m->conv - 1) * A->KV * sizeof(float));
        memset(A->ck, 0, (size_t)(m->conv - 1) * A->KV * sizeof(float));
        memset(A->cv, 0, (size_t)(m->conv - 1) * A->VV * sizeof(float));
    }
}

int sn_dn_ready(const SnDeltaNet* m) { return m && m->ready; }
int sn_dn_vocab(const SnDeltaNet* m) { return m ? m->vocab : 0; }

static void dn_layer(SnLayer* A, float* x) {
    const int D = A->D, NH = A->NH, HK = A->HK, HV = A->HV;
    const int KV = A->KV, VV = A->VV, F = A->F;
    float* sc = A->m->sc;
    float* h = sc;
    float* q = h + D;
    float* k = q + KV;
    float* v = k + KV;
    float* o = v + VV;
    float* beta = o + VV;
    float* kv = beta + NH;
    float* d = kv + (size_t)NH * HV;
    float* g = d + (size_t)NH * HV;
    float* u = g + F;

    rms_norm(h, x, A->an, D, A->eps);
    gemv_w(q, h, &A->qp);
    gemv_w(k, h, &A->kp);
    gemv_w(v, h, &A->vp);
    dconv_silu(q, q, g, A->cq, A->qc, KV, A->conv);
    dconv_silu(k, k, g, A->ck, A->kc, KV, A->conv);
    dconv_silu(v, v, g, A->cv, A->vc, VV, A->conv);

    const float qs = 1.0f / sqrtf((float)HK);
    for (int hh = 0; hh < NH; ++hh) {
        l2norm(q + (size_t)hh * HK, HK, qs);
        l2norm(k + (size_t)hh * HK, HK, 1.0f);
    }
    gemv(beta, h, A->bp, D, NH);
    for (int i = 0; i < NH; ++i) beta[i] = 1.0f / (1.0f + expf(-beta[i]));

    dn_attn(A, q, k, v, beta, o, kv, d);
    /* o_norm 逐 head 归一化 (参考实现按 (NH,HV) 的最后一维算) */
    for (int hh = 0; hh < NH; ++hh)
        rms_norm(o + (size_t)hh * HV, o + (size_t)hh * HV, A->on, HV, A->eps);
    gemv_w(h, o, &A->op);
    for (int i = 0; i < D; ++i) x[i] += h[i];

    rms_norm(h, x, A->mn, D, A->eps);
    gemv_w(g, h, &A->gp);
    gemv_w(u, h, &A->up);
    for (int i = 0; i < F; ++i) g[i] = silu(g[i]) * u[i];
    gemv_w(h, g, &A->dp);
    for (int i = 0; i < D; ++i) x[i] += h[i];
}

static void dn_forward(SnDeltaNet* m, int token, float* logits) {
    const int D = m->D;
    float* x = m->x;
    memcpy(x, m->embed + (size_t)token * D, (size_t)D * sizeof(float));
    for (int l = 0; l < m->L; ++l) dn_layer(&m->layers[l], x);
    rms_norm(x, x, m->ln_f, D, m->eps);
    if (m->head.w || m->head.is_q4)
        gemv_w(logits, x, &m->head);
    else
        gemv(logits, x, m->embed, D, m->vocab);   /* tied: 复用嵌入表 */
}

void sn_dn_step(SnDeltaNet* m, int token, float* logits) {
    if (!m || !m->ready) return;
    dn_forward(m, token, logits);
}

void sn_dn_prefill(SnDeltaNet* m, const int* tokens, int n, float* logits) {
    for (int i = 0; i < n; ++i) sn_dn_step(m, tokens[i], logits);
}

/* 跑完第 layer 层后把主流 x 拷出 (D 个 float); layer<0 表示跑完全部层。
 * 用于与参考实现逐层对表——层数一多, 只看最终 logits 定位不到是哪层偏的。 */
void sn_dn_dbg(SnDeltaNet* m, int token, int layer, float* out) {
    if (!m || !m->ready) return;
    const int D = m->D;
    memcpy(m->x, m->embed + (size_t)token * D, (size_t)D * sizeof(float));
    const int n = layer < 0 ? m->L : layer + 1;
    for (int l = 0; l < n && l < m->L; ++l) dn_layer(&m->layers[l], m->x);
    memcpy(out, m->x, (size_t)D * sizeof(float));
}