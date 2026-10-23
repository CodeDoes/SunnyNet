/* SunnyNet Runtime: HGRN2 整模型单步解码 (AVX2/FMA). */
#include "hgrn2.h"

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
/* 权重文件映射. 1.3B 的 fp32 blob 有 5.4GB, 复制进堆要同等物理内存,
 * 机器装不下两份; 映射后由页缓存共享, 只有真正读到的页占内存。 */
typedef struct { HANDLE fh, mh; const void* p; size_t n; } SnMap;
static int sn_map_open(const char* path, SnMap* m) {
    wchar_t wp[4096];
    if (!MultiByteToWideChar(CP_UTF8, 0, path, -1, wp, 4096)) return 0;
    HANDLE fh = CreateFileW(wp, GENERIC_READ, FILE_SHARE_READ, NULL,
                            OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (fh == INVALID_HANDLE_VALUE) return 0;
    LARGE_INTEGER sz;
    if (!GetFileSizeEx(fh, &sz) || sz.QuadPart <= 0) {
        CloseHandle(fh);
        return 0;
    }
    HANDLE mh = CreateFileMappingW(fh, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!mh) { CloseHandle(fh); return 0; }
    const void* p = MapViewOfFile(mh, FILE_MAP_READ, 0, 0, 0);
    if (!p) { CloseHandle(mh); CloseHandle(fh); return 0; }
    m->fh = fh; m->mh = mh; m->p = p; m->n = (size_t)sz.QuadPart;
    return 1;
}
static void sn_map_close(SnMap* m) {
    if (m->p) UnmapViewOfFile(m->p);
    if (m->mh) CloseHandle(m->mh);
    if (m->fh) CloseHandle(m->fh);
    m->p = NULL; m->mh = NULL; m->fh = NULL; m->n = 0;
}
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
static FILE* sn_fopen(const char* path, const char* mode) { return fopen(path, mode); }
typedef struct { int fd; const void* p; size_t n; } SnMap;
static int sn_map_open(const char* path, SnMap* m) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return 0;
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size <= 0) { close(fd); return 0; }
    void* p = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (p == MAP_FAILED) { close(fd); return 0; }
    m->fd = fd; m->p = p; m->n = (size_t)st.st_size;
    return 1;
}
static void sn_map_close(SnMap* m) {
    if (m->p) munmap((void*)m->p, m->n);
    if (m->fd >= 0) close(m->fd);
    m->p = NULL; m->fd = -1; m->n = 0;
}
#endif

void sn_hg_set_threads(int n) { if (n > 0) omp_set_num_threads(n); }
int sn_hg_get_threads(void) { return omp_get_max_threads(); }

#define SN_HG_MAGIC 0x32474853   /* "SHG2" */
#define SN_HG_VERSION 1

struct SnHGRN2;

typedef struct SnLayer {
    int D, NH, HK, HV, KD, VD, F, idx;
    float eps, scale;
    const float *lb;                  /* lower bound (D,) */
    const float *an, *mn;             /* attn_norm / mlp_norm (D,) */
    const float *qp, *fp, *ip, *op;   /* (out,in) 行主序 */
    const float *gn;                  /* g_norm (D,) */
    const float *mgp, *mup, *mdp;     /* MLP: (F,D) (F,D) (D,F) */
    float* S;                         /* (NH, HK, HV) */
    struct SnHGRN2* m;
} SnLayer;

struct SnHGRN2 {
    int D, NH, HK, HV, KD, VD, F, L, vocab, tied_head, use_lb;
    float eps, scale;
    const float* embed;
    const float* ln_f;
    const float* head;
    SnMap map;
    SnLayer* layers;
    float* x;
    float* sc;
    int ready;
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
    /* 写法与标量路径一致: 用 1/sqrt 广播, 不逐元素算 rsqrt */
    const float rinv = 1.0f / sqrtf(ss / (float)n + eps);
    for (i = 0; i + 8 <= n; i += 8)
        _mm256_storeu_ps(y + i, _mm256_mul_ps(
            _mm256_mul_ps(_mm256_loadu_ps(x + i), _mm256_set1_ps(rinv)),
            _mm256_loadu_ps(g + i)));
    for (; i < n; ++i) y[i] = x[i] * rinv * g[i];
}

static inline float silu(float x) { return x / (1.0f + expf(-x)); }

/* log(sigmoid(x)) = -softplus(-x); 直接 exp(-x) 在 x<<0 时溢出。 */
static inline float logsigmoid(float x) {
    const float a = -x;
    const float aa = fabsf(a);
    return -(log1pf(expf(-aa)) + (a > 0.0f ? a : 0.0f));
}

/* 状态更新 + 读出. 按 head 并行, head 内串行遍历 key (同一 head 的 K 行都写 o[h,:])。 */
static void hg_scan(SnLayer* A, const float* q, const float* k, const float* v,
                    const float* decay, float* o) {
    const int NH = A->NH, HK = A->HK, HV = A->HV;
    const float scale = A->scale;
    #pragma omp parallel for schedule(static)
    for (int h = 0; h < NH; ++h) {
        float* oh = o + (size_t)h * HV;
        const float* vh = v + (size_t)h * HV;
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

SnHGRN2* sn_hg_load(const char* path) {
    FILE* f = sn_fopen(path, "rb");
    if (!f) return NULL;
    int32_t h[8];
    if (!rd_i32(f, h, 8) || h[0] != SN_HG_MAGIC || h[1] != SN_HG_VERSION) {
        fclose(f);
        return NULL;
    }
    const int L = h[2], D = h[3], NH = h[4], HK = h[5], HV = h[6];
    float hf[4];
    if (fread(hf, sizeof(float), 4, f) != 4) { fclose(f); return NULL; }
    int32_t meta[5];
    if (!rd_i32(f, meta, 5)) { fclose(f); return NULL; }   /* vocab,F,tied,-,- */

    /* 张量流起始偏移 68 = 8i + 4f + 5i, 与 _export_hgrn2.py 的头部同步。
 * 不要用 fseek/ftell 探文件长度: mingw 的 long 是 32 位, 5.4GB 的 blob 会溢出。 */
    SnHGRN2* m = (SnHGRN2*)calloc(1, sizeof(SnHGRN2));
    if (!m) return NULL;
    /* 权重驻留用文件映射: 5.4GB fp32 复制进堆会耗尽物理内存 */
    if (!sn_map_open(path, &m->map)) { free(m); return NULL; }
    const unsigned char* buf = (const unsigned char*)m->map.p;
    m->layers = (SnLayer*)calloc((size_t)L, sizeof(SnLayer));
    if (!m->layers) { sn_map_close(&m->map); free(m); return NULL; }
    m->L = L; m->D = D; m->NH = NH; m->HK = HK; m->HV = HV;
    m->KD = NH * HK; m->VD = NH * HV; m->F = meta[1];
    m->vocab = meta[0]; m->tied_head = meta[2];
    m->eps = hf[0]; m->scale = hf[1]; m->use_lb = (hf[3] != 0.0f);

    Cur c = { buf + 68 };   /* 8i + 4f + 5i 头部, 与导出器 _export_hgrn2.py 同步 */
    for (int l = 0; l < L; ++l) {
        SnLayer* A = &m->layers[l];
        A->m = m;
        A->idx = l;
        A->D = D; A->NH = NH; A->HK = HK; A->HV = HV;
        A->KD = m->KD; A->VD = m->VD; A->F = m->F;
        A->eps = m->eps; A->scale = m->scale;
        /* 顺序由导出器 _export_hgrn2.py 决定, 两处必须同步改 */
        A->lb = take_f32(&c, D);
        A->an = take_f32(&c, D);
        A->qp = take_f32(&c, (size_t)D * D);
        A->fp = take_f32(&c, (size_t)D * D);
        A->ip = take_f32(&c, (size_t)D * D);
        A->op = take_f32(&c, (size_t)D * D);
        A->gn = take_f32(&c, D);
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
        if (!A->S) { sn_hg_free(m); return NULL; }
    }
    /* 暂存: h(D) q(KD) f(KD) g(KD) k(KD) i(VD) o(VD) mg(F) mu(F) */
    const size_t scn = (size_t)m->D + 4 * (size_t)m->KD + 2 * (size_t)m->VD
                       + 2 * (size_t)m->F + 32;
    m->x = (float*)calloc((size_t)D, sizeof(float));
    m->sc = (float*)calloc(scn, sizeof(float));
    if (!m->x || !m->sc) { sn_hg_free(m); return NULL; }
    m->ready = 1;
    return m;
}

void sn_hg_free(SnHGRN2* m) {
    if (!m) return;
    if (m->layers) {
        for (int l = 0; l < m->L; ++l) free(m->layers[l].S);
        free(m->layers);
    }
    free(m->x);
    free(m->sc);
    sn_map_close(&m->map);
    free(m);
}

void sn_hg_reset(SnHGRN2* m) {
    if (!m) return;
    for (int l = 0; l < m->L; ++l)
        memset(m->layers[l].S, 0,
               (size_t)m->NH * m->HK * m->HV * sizeof(float));
}

int sn_hg_ready(const SnHGRN2* m) { return m && m->ready; }
int sn_hg_vocab(const SnHGRN2* m) { return m ? m->vocab : 0; }

static void probe_dump(SnHGRN2* m, int stage, const float* src, int n) {
    if (m->probe && m->probe_stage == stage)
        memcpy(m->probe, src, (size_t)n * sizeof(float));
}

static void hg_layer(SnLayer* A, float* x) {
    const int D = A->D, NH = A->NH, HK = A->HK, HV = A->HV;
    const int KD = A->KD, VD = A->VD, F = A->F;
    SnHGRN2* m = A->m;
    float* sc = m->sc;
    float* h = sc;
    float* q = h + D;
    float* f = q + KD;
    float* g = f + KD;
    float* k = g + KD;
    float* v = k + KD;
    float* o = v + VD;
    float* mg = o + VD;
    float* mu = mg + F;

    rms_norm(h, x, A->an, D, A->eps);
    probe_dump(m, 0, h, D);

    gemv(q, h, A->qp, D, KD);
    for (int i = 0; i < KD; ++i) q[i] = silu(q[i]);   /* HGRN2 对 q 过 silu */
    probe_dump(m, 1, q, KD);

    gemv(f, h, A->fp, D, KD);
    gemv(v, h, A->ip, D, VD);
    probe_dump(m, 2, f, KD);
    probe_dump(m, 6, v, VD);

    /* g = logaddexp(lb, log1p(-lb) + logsigmoid(f)); k = 1 - exp(g)
     * 第 0 层 lb=0 => g = logsigmoid(f)。lb 存在则只有 l>0 参与混合。 */
    if (m->use_lb && A->idx > 0) {
        for (int i = 0; i < KD; ++i) {
            const float lb = A->lb[i];
            float gi = logsigmoid(f[i]);
            if (lb > 0.0f) {
                const float lg = logf(lb);
                const float u = log1pf(-lb) + gi;
                gi = (lg > u ? lg + log1pf(expf(u - lg))
                             : u + log1pf(expf(lg - u)));
            }
            g[i] = gi;
        }
    } else {
        for (int i = 0; i < KD; ++i) g[i] = logsigmoid(f[i]);
    }
    probe_dump(m, 3, g, KD);
    for (int i = 0; i < KD; ++i) g[i] = expf(g[i]);
    probe_dump(m, 4, g, KD);
    for (int i = 0; i < KD; ++i) k[i] = 1.0f - g[i];
    probe_dump(m, 5, k, KD);

    memset(o, 0, (size_t)VD * sizeof(float));
    hg_scan(A, q, k, v, g, o);
    probe_dump(m, 7, o, VD);

    /* 无输出门: 在 value 维整体 RMSNorm 后接 o_proj */
    rms_norm(o, o, A->gn, VD, A->eps);
    probe_dump(m, 8, o, VD);
    gemv(h, o, A->op, VD, D);
    probe_dump(m, 9, h, D);
    for (int i = 0; i < D; ++i) x[i] += h[i];

    rms_norm(h, x, A->mn, D, A->eps);
    gemv(mg, h, A->mgp, D, F);
    gemv(mu, h, A->mup, D, F);
    for (int i = 0; i < F; ++i) mg[i] = silu(mg[i]) * mu[i];
    gemv(h, mg, A->mdp, F, D);
    for (int i = 0; i < D; ++i) x[i] += h[i];
}

static void hg_forward(SnHGRN2* m, int token, float* logits) {
    const int D = m->D;
    float* x = m->x;
    memcpy(x, m->embed + (size_t)token * D, (size_t)D * sizeof(float));
    for (int l = 0; l < m->L; ++l) hg_layer(&m->layers[l], x);
    rms_norm(x, x, m->ln_f, D, m->eps);
    gemv(logits, x, m->head ? m->head : m->embed, D, m->vocab);
}

void sn_hg_step(SnHGRN2* m, int token, float* logits) {
    if (!m || !m->ready) return;
    hg_forward(m, token, logits);
}

void sn_hg_prefill(SnHGRN2* m, const int* tokens, int n, float* logits) {
    for (int i = 0; i < n; ++i) sn_hg_step(m, tokens[i], logits);
}

void sn_hg_dbg(SnHGRN2* m, int token, int layer, float* out) {
    if (!m || !m->ready) return;
    const int D = m->D;
    memcpy(m->x, m->embed + (size_t)token * D, (size_t)D * sizeof(float));
    const int n = layer < 0 ? m->L : layer + 1;
    for (int l = 0; l < n && l < m->L; ++l) hg_layer(&m->layers[l], m->x);
    memcpy(out, m->x, (size_t)D * sizeof(float));
}

void sn_hg_probe(SnHGRN2* m, int token, int layer, int stage, float* out) {
    if (!m || !m->ready || layer < 0 || layer >= m->L) return;
    m->probe = out;
    m->probe_stage = stage;
    memcpy(m->x, m->embed + (size_t)token * m->D, (size_t)m->D * sizeof(float));
    hg_layer(&m->layers[layer], m->x);
    m->probe = NULL;
}