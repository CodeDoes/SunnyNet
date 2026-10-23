/* SunnyNet Runtime: GLA 整模型单步解码 (AVX2/FMA). */
#include "gla.h"

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

void sn_gla_set_threads(int n) { if (n > 0) omp_set_num_threads(n); }
int sn_gla_get_threads(void) { return omp_get_max_threads(); }

#define SN_GLA_MAGIC 0x414C4753   /* "SGLA" */
#define SN_GLA_VERSION 1

struct SnGLA;

typedef struct SnLayer {
    int D, NH, HK, HV, GR, KD, VD, F, idx;
    float eps, scale, gnorm;
    const float *an, *mn;             /* attn_norm / mlp_norm (D,) */
    const float *qp, *kp, *vp, *op;   /* (out,in) 行主序 */
    const float *g0, *g1w, *g1b;      /* gk 低秩: (GR,D) (KD,GR) (KD,) */
    const float *gp, *gn;             /* g_proj (VD,D), g_norm_swish_gate (HV,) */
    const float *mgp, *mup, *mdp;     /* MLP: (F,D) (F,D) (D,F) */
    float* S;                         /* (NH, HK, HV) */
    struct SnGLA* m;
} SnLayer;

struct SnGLA {
    int D, NH, HK, HV, GR, KD, VD, F, L, vocab, tied_head;
    float eps, scale, gnorm;
    int clamp_min_set;
    float clamp_min;
    const float* embed;
    const float* ln_f;
    const float* head;
    unsigned char* wbuf;
    SnLayer* layers;
    float* x;
    float* sc;
    int ready;
    /* 探针: 非 NULL 时 gla_layer 在指定 stage 把中间量拷出 */
    float* probe;
    int probe_stage;
};

static inline float hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    return _mm_cvtss_f32(lo);
}

/* y = W x. W 存 (out,in) 行主序, 按输出通道并行。 */
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

/* log(sigmoid(x)) = -softplus(-x); 直接 exp(-x) 在 x<<0 时溢出。 */
static inline float logsigmoid(float x) {
    const float a = -x;
    const float aa = fabsf(a);
    return -(log1pf(expf(-aa)) + (a > 0.0f ? a : 0.0f));
}

/* 状态更新 + 读出. 按 head 并行: 同一 head 的 K 个 key 行都写 o[h,:],
 * 故 head 内必须串行遍历 key, 不能按行并行。 */
static void gla_scan(SnLayer* A, const float* q, const float* k, const float* v,
                     const float* decay, float* o) {
    const int NH = A->NH, HK = A->HK, HV = A->HV;
    const float scale = A->scale;
    #pragma omp parallel for schedule(static)
    for (int h = 0; h < NH; ++h) {
        float* oh = o + (size_t)h * HV;
        const float* vh = v + (size_t)h * HV;   /* 每个 head 自己的 value 段 */
        memset(oh, 0, (size_t)HV * sizeof(float));
        for (int kk = 0; kk < HK; ++kk) {
            const int hp = h * HK + kk;
            float* row = A->S + (size_t)hp * HV;
            const float da = decay[hp];
            const float co = k[hp];
            int j = 0;
            for (; j + 8 <= HV; j += 8)
                _mm256_storeu_ps(row + j, _mm256_fmadd_ps(
                    _mm256_set1_ps(co), _mm256_loadu_ps(vh + j),
                    _mm256_mul_ps(_mm256_loadu_ps(row + j), _mm256_set1_ps(da))));
            for (; j < HV; ++j) row[j] = row[j] * da + co * vh[j];
            const float qs = q[hp] * scale;
            j = 0;
            for (; j + 8 <= HV; j += 8)
                _mm256_storeu_ps(oh + j, _mm256_fmadd_ps(
                    _mm256_set1_ps(qs), _mm256_loadu_ps(row + j),
                    _mm256_loadu_ps(oh + j)));
            for (; j < HV; ++j) oh[j] += qs * row[j];
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

static int rd_i32(FILE* f, int32_t* o, int n) {
    return (int)fread(o, sizeof(int32_t), (size_t)n, f) == n;
}

SnGLA* sn_gla_load(const char* path) {
    FILE* f = sn_fopen(path, "rb");
    if (!f) return NULL;
    int32_t h[8];
    if (!rd_i32(f, h, 8) || h[0] != SN_GLA_MAGIC || h[1] != SN_GLA_VERSION) {
        fclose(f);
        return NULL;
    }
    const int L = h[2], D = h[3], NH = h[4], HK = h[5], HV = h[6], GR = h[7];
    float hf[4];
    if (fread(hf, sizeof(float), 4, f) != 4) { fclose(f); return NULL; }
    int32_t meta[5];
    if (!rd_i32(f, meta, 5)) { fclose(f); return NULL; }   /* vocab,F,tied,clamp_set,clamp */

    const long pos = ftell(f);
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    const size_t nbytes = (size_t)(ftell(f) - pos);
    unsigned char* buf = (unsigned char*)malloc(nbytes);
    if (!buf) { fclose(f); return NULL; }
    fseek(f, pos, SEEK_SET);
    if (fread(buf, 1, nbytes, f) != nbytes) { free(buf); fclose(f); return NULL; }
    fclose(f);

    SnGLA* m = (SnGLA*)calloc(1, sizeof(SnGLA));
    if (!m) { free(buf); return NULL; }
    m->layers = (SnLayer*)calloc((size_t)L, sizeof(SnLayer));
    if (!m->layers) { free(buf); free(m); return NULL; }
    m->L = L; m->D = D; m->NH = NH; m->HK = HK; m->HV = HV; m->GR = GR;
    m->KD = NH * HK; m->VD = NH * HV; m->F = meta[1];
    m->vocab = meta[0]; m->tied_head = meta[2];
    m->clamp_min_set = meta[3]; m->clamp_min = hf[3];
    m->eps = hf[0]; m->scale = hf[1]; m->gnorm = hf[2];

    Cur c = { buf };
    for (int l = 0; l < L; ++l) {
        SnLayer* A = &m->layers[l];
        A->m = m;
        A->D = D; A->NH = NH; A->HK = HK; A->HV = HV; A->GR = GR;
        A->KD = m->KD; A->VD = m->VD; A->F = m->F;
        A->eps = m->eps; A->scale = m->scale; A->gnorm = m->gnorm;
        /* 顺序由导出器 _export_gla.py 决定, 两处必须同步改 */
        A->an = take_f32(&c, D);
        A->qp = take_f32(&c, (size_t)m->KD * D);
        A->kp = take_f32(&c, (size_t)m->KD * D);
        A->vp = take_f32(&c, (size_t)m->VD * D);
        A->op = take_f32(&c, (size_t)D * m->VD);
        A->g0 = take_f32(&c, (size_t)GR * D);
        A->g1w = take_f32(&c, (size_t)m->KD * GR);
        A->g1b = take_f32(&c, m->KD);
        A->gp = take_f32(&c, (size_t)m->VD * D);
        A->gn = take_f32(&c, HV);
        A->mn = take_f32(&c, D);
        A->mgp = take_f32(&c, (size_t)m->F * D);
        A->mup = take_f32(&c, (size_t)m->F * D);
        A->mdp = take_f32(&c, (size_t)D * m->F);
    }
    m->embed = take_f32(&c, (size_t)m->vocab * D);
    m->ln_f = take_f32(&c, D);
    m->head = m->tied_head ? NULL : take_f32(&c, (size_t)m->vocab * D);

    for (int l = 0; l < L; ++l) {
        SnLayer* A = &m->layers[l];
        A->S = (float*)calloc((size_t)NH * HK * HV, sizeof(float));
        if (!A->S) { sn_gla_free(m); return NULL; }
    }
    /* 暂存: h(D) q(KD) k(KD) v(VD) o(VD) gk(KD) gk2(GR) gg(VD) mg(F) mu(F) */
    const size_t scn = (size_t)m->D + 2 * (size_t)m->KD + 3 * (size_t)m->VD
                       + (size_t)m->KD + (size_t)m->GR + 2 * (size_t)m->F + 32;
    m->x = (float*)calloc((size_t)D, sizeof(float));
    m->sc = (float*)calloc(scn, sizeof(float));
    if (!m->x || !m->sc) { sn_gla_free(m); return NULL; }
    m->wbuf = buf;
    m->ready = 1;
    return m;
}

void sn_gla_free(SnGLA* m) {
    if (!m) return;
    if (m->layers) {
        for (int l = 0; l < m->L; ++l) free(m->layers[l].S);
        free(m->layers);
    }
    free(m->x);
    free(m->sc);
    free(m->wbuf);
    free(m);
}

void sn_gla_reset(SnGLA* m) {
    if (!m) return;
    for (int l = 0; l < m->L; ++l)
        memset(m->layers[l].S, 0,
               (size_t)m->NH * m->HK * m->HV * sizeof(float));
}

int sn_gla_ready(const SnGLA* m) { return m && m->ready; }
int sn_gla_vocab(const SnGLA* m) { return m ? m->vocab : 0; }

/* 探针辅助: stage 命中就把 src 的 n 个 float 拷进探针缓冲。
 * stage 编号见 sn_gla_probe 的注释。 */
static void probe_dump(SnGLA* m, int stage, const float* src, int n) {
    if (m->probe && m->probe_stage == stage)
        memcpy(m->probe, src, (size_t)n * sizeof(float));
}

static void gla_layer(SnLayer* A, float* x) {
    const int D = A->D, NH = A->NH, HK = A->HK, HV = A->HV;
    const int KD = A->KD, VD = A->VD, GR = A->GR, F = A->F;
    float* sc = A->m->sc;
    float* h = sc;
    float* q = h + D;
    float* k = q + KD;
    float* v = k + KD;
    float* o = v + VD;
    float* gk = o + VD;
    float* gk2 = gk + KD;
    float* gg = gk2 + GR;
    float* mg = gg + VD;
    float* mu = mg + F;

    rms_norm(h, x, A->an, D, A->eps);
    gemv(q, h, A->qp, D, KD);
    gemv(k, h, A->kp, D, KD);
    gemv(v, h, A->vp, D, VD);
    gemv(gg, h, A->gp, D, VD);
    /* gk = logsigmoid(W1 (W0 h) + b) / normalizer, 逐 key 维 */
    gemv(gk2, h, A->g0, D, GR);
    gemv(gk, gk2, A->g1w, GR, KD);
    const float inv = 1.0f / A->gnorm;
    for (int i = 0; i < KD; ++i) {
        float g = logsigmoid(gk[i] + A->g1b[i]) * inv;
        if (A->m->clamp_min_set && g < A->m->clamp_min) g = A->m->clamp_min;
        gk[i] = expf(g);
    }
    probe_dump(A->m, 0, h, D);      /* attn_norm 输出 */
    probe_dump(A->m, 1, q, KD);     /* q_proj 输出 */
    probe_dump(A->m, 2, k, KD);
    probe_dump(A->m, 3, v, VD);
    probe_dump(A->m, 4, gk, KD);    /* exp 后的衰减 */
    probe_dump(A->m, 5, gg, VD);    /* g_proj 输出 */

    memset(o, 0, (size_t)VD * sizeof(float));
    gla_scan(A, q, k, v, gk, o);
    probe_dump(A->m, 6, o, VD);     /* 读出, 门之前的 o */

    /* 输出门: 逐 head 在 head_v_dim 上 RMSNorm, 再乘 silu(g_proj) */
    for (int hh = 0; hh < NH; ++hh)
        rms_norm(o + (size_t)hh * HV, o + (size_t)hh * HV, A->gn, HV, A->eps);
    probe_dump(A->m, 7, o, VD);     /* RMSNorm 之后 */
    for (int i = 0; i < VD; ++i) o[i] *= silu(gg[i]);
    probe_dump(A->m, 8, o, VD);     /* 门之后 */
    gemv(h, o, A->op, VD, D);
    probe_dump(A->m, 9, h, D);      /* o_proj 输出 */
    for (int i = 0; i < D; ++i) x[i] += h[i];

    rms_norm(h, x, A->mn, D, A->eps);
    gemv(mg, h, A->mgp, D, F);
    gemv(mu, h, A->mup, D, F);
    for (int i = 0; i < F; ++i) mg[i] = silu(mg[i]) * mu[i];
    gemv(h, mg, A->mdp, F, D);
    for (int i = 0; i < D; ++i) x[i] += h[i];
}

static void gla_forward(SnGLA* m, int token, float* logits) {
    const int D = m->D;
    float* x = m->x;
    memcpy(x, m->embed + (size_t)token * D, (size_t)D * sizeof(float));
    for (int l = 0; l < m->L; ++l) gla_layer(&m->layers[l], x);
    rms_norm(x, x, m->ln_f, D, m->eps);
    gemv(logits, x, m->head ? m->head : m->embed, D, m->vocab);
}

void sn_gla_step(SnGLA* m, int token, float* logits) {
    if (!m || !m->ready) return;
    gla_forward(m, token, logits);
}

void sn_gla_prefill(SnGLA* m, const int* tokens, int n, float* logits) {
    for (int i = 0; i < n; ++i) sn_gla_step(m, tokens[i], logits);
}

/* 跑完第 layer 层后把主流 x 拷出; layer<0 表示跑完全部层。
 * dbg 会推进状态, 逐层对表时每次调用前须 reset。 */
void sn_gla_dbg(SnGLA* m, int token, int layer, float* out) {
    if (!m || !m->ready) return;
    const int D = m->D;
    memcpy(m->x, m->embed + (size_t)token * D, (size_t)D * sizeof(float));
    const int n = layer < 0 ? m->L : layer + 1;
    for (int l = 0; l < n && l < m->L; ++l) gla_layer(&m->layers[l], m->x);
    memcpy(out, m->x, (size_t)D * sizeof(float));
}

/* 层内阶段探针: 跑第 layer 层的第 token 个 token, 在 stage 处把中间量拷进 out。
 * stage: 0 attn_norm | 1 q | 2 k | 3 v | 4 exp(gk) | 5 g_proj
 *        6 读出(门之前) | 7 逐 head RMSNorm 后 | 8 门后 | 9 o_proj
 * 调用前须 reset。 */
void sn_gla_probe(SnGLA* m, int token, int layer, int stage, float* out) {
    if (!m || !m->ready || layer < 0 || layer >= m->L) return;
    m->probe = out;
    m->probe_stage = stage;
    memcpy(m->x, m->embed + (size_t)token * m->D, (size_t)m->D * sizeof(float));
    gla_layer(&m->layers[layer], m->x);
    m->probe = NULL;
}