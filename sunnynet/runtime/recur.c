/* SunnyNet Runtime: 通用一阶线性递推内核实现 (AVX2/FMA + OpenMP). */
#include "recur.h"

#include <immintrin.h>
#include <math.h>
#include <stddef.h>
#include <omp.h>

void sn_set_threads(int n) { omp_set_num_threads(n); }
int sn_threads(void) { return omp_get_max_threads(); }

/* a, b: (B,T,N) 连续. 沿 N 向量化, T 串行 (依赖链). */
void sn_recur_scan(const float* a, const float* b, const float* s0,
                   float* out, float* send, int B, int T, int N, int gated) {
    const __m256 one = _mm256_set1_ps(1.0f);
    #pragma omp parallel for schedule(static)
    for (int bb = 0; bb < B; ++bb) {
        const float* ab = a + (size_t)bb * T * N;
        const float* bbb = b + (size_t)bb * T * N;
        float* ob = out + (size_t)bb * T * N;
        const float* s0b = s0 + (size_t)bb * N;
        float* seb = send + (size_t)bb * N;
        int d = 0;
        for (; d + 8 <= N; d += 8) {
            __m256 s = _mm256_loadu_ps(s0b + d);
            for (int t = 0; t < T; ++t) {
                const size_t off = (size_t)t * N + d;
                __m256 av = _mm256_loadu_ps(ab + off);
                __m256 bv = _mm256_loadu_ps(bbb + off);
                if (gated) bv = _mm256_mul_ps(_mm256_sub_ps(one, av), bv);
                s = _mm256_fmadd_ps(av, s, bv);
                _mm256_storeu_ps(ob + off, s);
            }
            _mm256_storeu_ps(seb + d, s);
        }
        for (; d < N; ++d) {
            float s = s0b[d];
            for (int t = 0; t < T; ++t) {
                const size_t off = (size_t)t * N + d;
                const float av = ab[off];
                const float bv = gated ? (1.0f - av) * bbb[off] : bbb[off];
                s = av * s + bv;
                ob[off] = s;
            }
            seb[d] = s;
        }
    }
}

/* ds_t = dout_t + a_{t+1} * ds_{t+1} */
void sn_recur_bwd(const float* a, const float* b, const float* s0,
                  const float* out, const float* dout,
                  float* da, float* db, float* ds0,
                  int B, int T, int N, int gated) {
    const __m256 one = _mm256_set1_ps(1.0f);
    #pragma omp parallel for schedule(static)
    for (int bb = 0; bb < B; ++bb) {
        const float* ab = a + (size_t)bb * T * N;
        const float* bbb = b + (size_t)bb * T * N;
        const float* ob = out + (size_t)bb * T * N;
        const float* dob = dout + (size_t)bb * T * N;
        const float* s0b = s0 + (size_t)bb * N;
        float* dab = da + (size_t)bb * T * N;
        float* dbb = db + (size_t)bb * T * N;
        float* d0b = ds0 + (size_t)bb * N;
        int d = 0;
        for (; d + 8 <= N; d += 8) {
            __m256 ds = _mm256_setzero_ps();
            for (int t = T - 1; t >= 0; --t) {
                const size_t off = (size_t)t * N + d;
                ds = _mm256_add_ps(_mm256_loadu_ps(dob + off), ds);
                const __m256 av = _mm256_loadu_ps(ab + off);
                const __m256 sprev = (t > 0)
                    ? _mm256_loadu_ps(ob + off - N)
                    : _mm256_loadu_ps(s0b + d);
                if (gated) {
                    const __m256 uv = _mm256_loadu_ps(bbb + off);
                    _mm256_storeu_ps(dab + off, _mm256_mul_ps(ds, _mm256_sub_ps(sprev, uv)));
                    _mm256_storeu_ps(dbb + off, _mm256_mul_ps(ds, _mm256_sub_ps(one, av)));
                } else {
                    _mm256_storeu_ps(dab + off, _mm256_mul_ps(ds, sprev));
                    _mm256_storeu_ps(dbb + off, ds);
                }
                ds = _mm256_mul_ps(ds, av);
            }
            _mm256_storeu_ps(d0b + d, ds);
        }
        for (; d < N; ++d) {
            float ds = 0.0f;
            for (int t = T - 1; t >= 0; --t) {
                const size_t off = (size_t)t * N + d;
                ds += dob[off];
                const float av = ab[off];
                const float sprev = (t > 0) ? ob[off - N] : s0b[d];
                if (gated) {
                    dab[off] = ds * (sprev - bbb[off]);
                    dbb[off] = ds * (1.0f - av);
                } else {
                    dab[off] = ds * sprev;
                    dbb[off] = ds;
                }
                ds *= av;
            }
            d0b[d] = ds;
        }
    }
}

void sn_recur_step(const float* a, const float* b, float* s, int B, int N, int gated) {
    const __m256 one = _mm256_set1_ps(1.0f);
    const int work = B * (N >> 3);
    #pragma omp parallel for schedule(static) if(work >= 256)
    for (int bb = 0; bb < B; ++bb) {
        const float* ab = a + (size_t)bb * N;
        const float* bbb = b + (size_t)bb * N;
        float* sb = s + (size_t)bb * N;
        int d = 0;
        for (; d + 8 <= N; d += 8) {
            __m256 av = _mm256_loadu_ps(ab + d);
            __m256 bv = _mm256_loadu_ps(bbb + d);
            __m256 sv = _mm256_loadu_ps(sb + d);
            if (gated) bv = _mm256_mul_ps(_mm256_sub_ps(one, av), bv);
            _mm256_storeu_ps(sb + d, _mm256_fmadd_ps(av, sv, bv));
        }
        for (; d < N; ++d) {
            const float av = ab[d];
            const float bv = gated ? (1.0f - av) * bbb[d] : bbb[d];
            sb[d] = av * sb[d] + bv;
        }
    }
}

/* 对称 INT8: scale = max|x| / 127, q = round(x / scale) 截断到 [-127,127]. */
void sn_quant_i8(const float* x, signed char* q, float* scale,
                 int B, int N, int block) {
    if (block <= 0) block = N;
    const int nb = (N + block - 1) / block;
    #pragma omp parallel for schedule(static)
    for (int bb = 0; bb < B; ++bb) {
        const float* xb = x + (size_t)bb * N;
        signed char* qb = q + (size_t)bb * N;
        float* scb = scale + (size_t)bb * nb;
        for (int k = 0; k < nb; ++k) {
            const int lo = k * block, hi = (k + 1) * block < N ? (k + 1) * block : N;
            float amax = 0.0f;
            for (int i = lo; i < hi; ++i) {
                const float v = fabsf(xb[i]);
                if (v > amax) amax = v;
            }
            const float sc = amax > 0.0f ? amax / 127.0f : 1.0f;
            const float inv = sc > 0.0f ? 1.0f / sc : 0.0f;
            scb[k] = sc;
            for (int i = lo; i < hi; ++i) {
                float t = nearbyintf(xb[i] * inv);
                if (t > 127.0f) t = 127.0f;
                if (t < -127.0f) t = -127.0f;
                qb[i] = (signed char)t;
            }
        }
    }
}

void sn_dequant_i8(const signed char* q, const float* scale, float* x,
                   int B, int N, int block) {
    if (block <= 0) block = N;
    const int nb = (N + block - 1) / block;
    #pragma omp parallel for schedule(static)
    for (int bb = 0; bb < B; ++bb) {
        const signed char* qb = q + (size_t)bb * N;
        float* xb = x + (size_t)bb * N;
        const float* scb = scale + (size_t)bb * nb;
        for (int k = 0; k < nb; ++k) {
            const int lo = k * block, hi = (k + 1) * block < N ? (k + 1) * block : N;
            const float sc = scb[k];
            for (int i = lo; i < hi; ++i) xb[i] = (float)qb[i] * sc;
        }
    }
}