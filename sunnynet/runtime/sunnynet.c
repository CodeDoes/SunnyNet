/* SunnyNet Runtime: B=1 decode step, whole model in C.
 *
 * Fixed default path only: decay_gate on, circulant on, local taps on,
 * current on, residual on, state_norm off, state_mul off. The exporter
 * refuses other ablations.
 *
 * Matmul weights are stored transposed (in, out) so one AVX2 load feeds 8
 * output rows; no intermediate tensors are materialised.
 */
#include "sunnynet.h"

#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#ifdef _WIN32
#include <windows.h>
/* mingw 的 fopen 走 ANSI 代码页, 非 ASCII 路径 (中文用户名) 打不开. */
static FILE* sn_fopen(const char* path, const char* mode) {
    wchar_t wp[1024], wm[16];
    if (MultiByteToWideChar(CP_UTF8, 0, path, -1, wp, 1024) <= 0) return NULL;
    MultiByteToWideChar(CP_UTF8, 0, mode, -1, wm, 16);
    return _wfopen(wp, wm);
}
#else
#define sn_fopen fopen
#endif

#define SN_MAGIC 0x4E4E5355 /* "USNN" */

void sn_set_threads(int n) { if (n > 0) omp_set_num_threads(n); }
int sn_get_threads(void) { return omp_get_max_threads(); }

/* 物理核数. 解码是访存受限, 超线程不增加带宽, 反而因争用变慢
 * (p100mw 实测 6 核 26.7 ms vs 12 逻辑核 36.0 ms)。 */
int sn_recommended_threads(void) {
#ifdef _WIN32
    DWORD len = 0;
    GetLogicalProcessorInformation(NULL, &len);
    if (len == 0) return 0;
    SYSTEM_LOGICAL_PROCESSOR_INFORMATION* buf =
        (SYSTEM_LOGICAL_PROCESSOR_INFORMATION*)malloc(len);
    if (!buf) return 0;
    int n = 0;
    if (GetLogicalProcessorInformation(buf, &len)) {
        const DWORD cnt = len / sizeof(*buf);
        for (DWORD i = 0; i < cnt; ++i)
            if (buf[i].Relationship == RelationProcessorCore) ++n;
    }
    free(buf);
    return n;
#else
    FILE* f = fopen("/proc/cpuinfo", "r");
    if (!f) return 0;
    char line[256];
    int phys = -1, core = -1, n = 0;
    int seen[256][2];
    while (fgets(line, sizeof line, f)) {
        int v;
        if (sscanf(line, "physical id : %d", &v) == 1) phys = v;
        else if (sscanf(line, "core id : %d", &v) == 1) core = v;
        else if (line[0] == '\n') {
            if (core >= 0) {
                int dup = 0;
                for (int i = 0; i < n; ++i)
                    if (seen[i][0] == phys && seen[i][1] == core) { dup = 1; break; }
                if (!dup && n < 256) { seen[n][0] = phys; seen[n][1] = core; ++n; }
            }
            phys = core = -1;
        }
    }
    fclose(f);
    return n > 0 ? n : omp_get_num_procs();
#endif
}

/* 权重在内存中为行主序 (out, in): 输出通道 j 的权重连续。 */
typedef struct {
    int D, I, R, K;
    const float* ni_g;
    const float* ni_c;
    const float* nm_g;
    const float* nm_c;
    const float* dw_t;   /* (K,D) 行主序, 每 tap 一段连续 */
    const float* Vt;     /* V.weight 行主序 (R,D) */
    const float* Ut;     /* U.weight 行主序 (2D,R) */
    const float* Wz_t;   /* W_z 行主序 (I,D) */
    const float* Wc_t;   /* W_c 行主序 (D,I) */
    const float* Ws_t;   /* W_s_eff 行主序 (D,D) */
    const float* delta;  /* (D) */
    float a_tanh, b_gain;
    float* s;
    float* hist;
} SnLayer;

struct SnModel {
    int vocab, D, I, R, K, L, group, tied, clamp;
    float eps, p_floor;
    float* wbuf;
    SnLayer* layers;
    const float* no_g;
    const float* no_c;
    const float* embed;  /* (vocab,D) 查表用 */
    const float* head_t; /* (D,vocab) 输出投影用 */
    float* sc;           /* 单层工作区 */
    float* sc2;          /* 顶层工作区 (x,lout,nrm) */
    int ready;
};

/* y[out] = W_rowmajor (out,in) · x[in]. 输出通道之间无依赖, 按行切分并行。 */
static void gemv(float* y, const float* x, const float* W, int in, int out) {
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < out; ++j) {
        const float* row = W + (size_t)j * in;
        __m256 acc = _mm256_setzero_ps();
        int i = 0;
        for (; i + 8 <= in; i += 8)
            acc = _mm256_fmadd_ps(_mm256_loadu_ps(row + i),
                                  _mm256_loadu_ps(x + i), acc);
        float t[8];
        _mm256_storeu_ps(t, acc);
        float s = t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
        for (; i < in; ++i) s += row[i] * x[i];
        y[j] = s;
    }
}

/* 把导出器的 (in,out) 转置块翻成行主序 (out,in)。分块避免跨步抖动。 */
static void transpose_into(float* dst, const float* src, int in, int out) {
    const int BS = 32;
    #pragma omp parallel for schedule(static)
    for (int jb = 0; jb < out; jb += BS) {
        const int jn = (jb + BS <= out) ? BS : out - jb;
        for (int ib = 0; ib < in; ib += BS) {
            const int inn = (ib + BS <= in) ? BS : in - ib;
            for (int j = jb; j < jb + jn; ++j) {
                float* d = dst + (size_t)j * in + ib;
                const float* sr = src + (size_t)ib * out + j;
                for (int i = 0; i < inn; ++i) d[i] = sr[(size_t)i * out];
            }
        }
    }
}

/* 原地转置: 元素数不变, 后续区段偏移不受影响。 */
static void flip_inplace(float* region, int in, int out) {
    const size_t n = (size_t)in * out;
    float* tmp = (float*)malloc(sizeof(float) * n);
    if (!tmp) return;              /* 退化: 保持导出布局, 结果仍正确但更慢 */
    memcpy(tmp, region, sizeof(float) * n);
    transpose_into(region, tmp, in, out);
    free(tmp);
}

static inline float sgn(float x) { return 1.0f / (1.0f + expf(-x)); }

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

static void layer_step(SnModel* m, SnLayer* L, const float* x, float* out, float* dbg) {
    const int D = L->D, I = L->I, R = L->R, K = L->K, G = m->group;
    float* sc = m->sc;
    float* raw = sc;        /* D  */
    float* at = raw + D;    /* D  */
    float* v = at + D;      /* R  */
    float* q = v + R;       /* 2D */
    float* u = q + 2 * D;   /* D  */
    float* gate = u + D;    /* D  */
    float* zn = gate + D;   /* I  */
    float* yn = zn + I;     /* I  */
    float* sfb = yn + I;    /* D  */

    layernorm(raw, x, L->ni_g, L->ni_c, D, m->eps);

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

    gemv(v, at, L->Vt, D, R);
    gemv(q, v, L->Ut, R, 2 * D);
    memcpy(u, q, sizeof(float) * D);
    memcpy(gate, q + D, sizeof(float) * D);

    for (int i = 0; i < D; ++i) {
        const int base = (i / G) * G, off = i - base;
        u[i] *= sgn(gate[i] - gate[base + (off + 1) % G]) *
                sgn(gate[i] - gate[base + (off + 2) % G]);
    }

    /* sfb 依赖上一时刻状态, 必须在状态更新前算 */
    gemv(sfb, L->s, L->Ws_t, D, D);

    for (int i = 0; i < D; ++i) {
        float p = 1.0f - L->delta[i] * (1.0f - sgn(gate[i]));
        if (m->clamp && p < m->p_floor) p = m->p_floor;
        L->s[i] = p * L->s[i] + (1.0f - p) * u[i];
    }

    gemv(zn, u, L->Wz_t, D, I);
    for (int i = 0; i < I; ++i)
        zn[i] *= 1.0f + 0.5f * L->a_tanh / (1.0f + fabsf(zn[i]));
    layernorm(yn, zn, L->nm_g, L->nm_c, I, m->eps);

    gemv(out, yn, L->Wc_t, I, D);
    for (int i = 0; i < D; ++i) out[i] += L->b_gain * L->s[i] + sfb[i] + x[i];

    if (dbg) {
        float* t = dbg;
        memcpy(t, raw, sizeof(float) * D); t += D;
        memcpy(t, at, sizeof(float) * D); t += D;
        memcpy(t, v, sizeof(float) * R); t += R;
        memcpy(t, gate, sizeof(float) * D); t += D;
        memcpy(t, u, sizeof(float) * D); t += D;
        memcpy(t, L->s, sizeof(float) * D); t += D;
        memcpy(t, zn, sizeof(float) * I); t += I;
        memcpy(t, yn, sizeof(float) * I); t += I;
        memcpy(t, sfb, sizeof(float) * D); t += D;
        memcpy(t, out, sizeof(float) * D); t += D;
    }
}

int sn_ready(const SnModel* m) { return m && m->ready; }
int sn_vocab(const SnModel* m) { return m ? m->vocab : 0; }

void sn_reset(SnModel* m) {
    if (!m) return;
    for (int l = 0; l < m->L; ++l) {
        SnLayer* L = &m->layers[l];
        memset(L->s, 0, sizeof(float) * (size_t)L->D);
        memset(L->hist, 0, sizeof(float) * (size_t)(L->K > 1 ? L->K - 1 : 1) * L->D);
    }
}

void sn_free(SnModel* m) {
    if (!m) return;
    if (m->layers) {
        for (int l = 0; l < m->L; ++l) {
            free(m->layers[l].s);
            free(m->layers[l].hist);
        }
    }
    free(m->wbuf);
    free(m->layers);
    free(m->sc);
    free(m->sc2);
    free(m);
}

void sn_step(SnModel* m, int token, float* logits) {
    const int D = m->D;
    float* x = m->sc2;
    float* lout = x + D;
    float* nrm = lout + D;
    memcpy(x, m->embed + (size_t)token * D, sizeof(float) * D);
    for (int l = 0; l < m->L; ++l) {
        layer_step(m, &m->layers[l], x, lout, NULL);
        memcpy(x, lout, sizeof(float) * D);
    }
    layernorm(nrm, x, m->no_g, m->no_c, D, m->eps);
    gemv(logits, nrm, m->head_t, D, m->vocab);
}

void sn_dbg_layer(SnModel* m, int token, int layer, float* dbg) {
    const int D = m->D;
    float* x = m->sc2;
    float* lout = x + D;
    memcpy(x, m->embed + (size_t)token * D, sizeof(float) * D);
    for (int l = 0; l <= layer; ++l) {
        layer_step(m, &m->layers[l], x, lout, l == layer ? dbg : NULL);
        memcpy(x, lout, sizeof(float) * D);
    }
}

void sn_prefill(SnModel* m, const int* tokens, int n, float* logits) {
    for (int t = 0; t < n; ++t) sn_step(m, tokens[t], logits);
}

static int rd_i32(FILE* f, int32_t* o, int n) {
    return (int)fread(o, sizeof(int32_t), (size_t)n, f) == n;
}
static int rd_f32(FILE* f, float* o, size_t n) {
    return fread(o, sizeof(float), n, f) == n;
}

SnModel* sn_load(const char* path, int vocab, int dim, int inner, int rank,
                 int taps, int layers, int group, int tied) {
    FILE* f = sn_fopen(path, "rb");
    if (!f) return NULL;
    int32_t h[10];
    if (!rd_i32(f, h, 10) || h[0] != SN_MAGIC || h[1] != vocab || h[2] != dim ||
        h[3] != inner || h[4] != rank || h[5] != taps || h[6] != layers ||
        h[7] != group || h[8] != tied) {
        fclose(f);
        return NULL;
    }
    SnModel* m = (SnModel*)calloc(1, sizeof(SnModel));
    if (!m) { fclose(f); return NULL; }
    m->vocab = vocab; m->D = dim; m->I = inner; m->R = rank;
    m->K = taps; m->L = layers; m->group = group; m->tied = h[8];
    m->clamp = h[9];

    const size_t per_layer = (size_t)2 * dim + (size_t)2 * inner +
                             (size_t)taps * dim + (size_t)dim * rank +
                             (size_t)rank * 2 * dim + (size_t)dim * inner +
                             (size_t)inner * dim + (size_t)dim * dim + dim + 2;
    const size_t total = (size_t)layers * per_layer + 2 + 2 * (size_t)dim +
                         (size_t)vocab * dim + (size_t)dim * vocab;
    m->wbuf = (float*)malloc(sizeof(float) * total);
    m->layers = (SnLayer*)calloc((size_t)layers, sizeof(SnLayer));
    if (!m->wbuf || !m->layers) { sn_free(m); fclose(f); return NULL; }

    float* p = m->wbuf;
    for (int l = 0; l < layers; ++l) {
        SnLayer* L = &m->layers[l];
        L->D = dim; L->I = inner; L->R = rank; L->K = taps;
        L->ni_g = p; p += dim;
        L->ni_c = p; p += dim;
        L->nm_g = p; p += inner;
        L->nm_c = p; p += inner;
        L->dw_t = p; p += (size_t)taps * dim;
        L->Vt = p;   p += (size_t)dim * rank;
        L->Ut = p;   p += (size_t)rank * 2 * dim;
        L->Wz_t = p; p += (size_t)dim * inner;
        L->Wc_t = p; p += (size_t)inner * dim;
        L->Ws_t = p; p += (size_t)dim * dim;
        L->delta = p; p += dim;
        /* 标量 (a_tanh,b_gain) 与权重同块, 保持顺序一致 */
        float* scal = p; p += 2;
        (void)scal;
    }
    float* eps = p; p += 1;
    float* pfloor = p; p += 1;
    float* no_g = p; p += dim;
    float* no_c = p; p += dim;
    float* embed = p; p += (size_t)vocab * dim;
    float* head_t = p; p += (size_t)dim * vocab;

    if (!rd_f32(f, m->wbuf, (size_t)(p - m->wbuf))) {
        sn_free(m); fclose(f); return NULL;
    }
    fclose(f);

    m->eps = *eps;
    m->p_floor = *pfloor;
    m->no_g = no_g;
    m->no_c = no_c;
    m->embed = embed;
    m->head_t = head_t;

    float* q = m->wbuf;
    for (int l = 0; l < layers; ++l) {
        SnLayer* L = &m->layers[l];
        /* 导出块按导出时的逻辑形状 (in,out) 解释后翻转成 (out,in)。
         * 顺序: Vt=(D,R) Ut=(R,2D) Wz_t=(D,I) Wc_t=(I,D) Ws_t=(D,D)。 */
        flip_inplace((float*)L->Vt, dim, rank);
        flip_inplace((float*)L->Ut, rank, 2 * dim);
        flip_inplace((float*)L->Wz_t, dim, inner);
        flip_inplace((float*)L->Wc_t, inner, dim);
        flip_inplace((float*)L->Ws_t, dim, dim);

        q += 2 * dim + 2 * inner + (size_t)taps * dim + (size_t)dim * rank +
             (size_t)rank * 2 * dim + (size_t)dim * inner + (size_t)inner * dim +
             (size_t)dim * dim + dim;
        L->a_tanh = q[0];
        L->b_gain = q[1];
        q += 2;

        L->s = (float*)calloc((size_t)dim, sizeof(float));
        L->hist = (float*)calloc((size_t)(taps > 1 ? taps - 1 : 1) * dim, sizeof(float));
        if (!L->s || !L->hist) { sn_free(m); return NULL; }
    }
    flip_inplace((float*)m->head_t, dim, m->vocab);

    const size_t scn = (size_t)3 * dim + rank + (size_t)2 * dim + (size_t)2 * dim +
                       (size_t)2 * inner + dim;
    m->sc = (float*)calloc(scn, sizeof(float));
    m->sc2 = (float*)calloc((size_t)3 * dim, sizeof(float));
    if (!m->sc || !m->sc2) { sn_free(m); return NULL; }
    m->ready = 1;
    return m;
}