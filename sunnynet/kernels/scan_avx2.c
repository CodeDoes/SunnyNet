/* SunnyNet 扫描算子 AVX2 内核. s_t = p_t*s_{t-1} + (1-p_t)*nu_t, 逐通道独立. */
#include <immintrin.h>
#include <stddef.h>

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

/* p, nu: (B,T,N) 连续; s0: (B,N); out: (B,T,N); send: (B,N) 末态 */
EXPORT void scan_f32(const float* p, const float* nu, const float* s0,
                     float* out, float* send, int B, int T, int N) {
    const float one = 1.0f;
    for (int b = 0; b < B; ++b) {
        const float* pb = p + (size_t)b * T * N;
        const float* nb = nu + (size_t)b * T * N;
        float* ob = out + (size_t)b * T * N;
        const float* s0b = s0 + (size_t)b * N;
        int d = 0;
        for (; d + 8 <= N; d += 8) {
            __m256 s = _mm256_loadu_ps(s0b + d);
            const __m256 on = _mm256_set1_ps(one);
            for (int t = 0; t < T; ++t) {
                size_t off = (size_t)t * N + d;
                __m256 pv = _mm256_loadu_ps(pb + off);
                __m256 nv = _mm256_loadu_ps(nb + off);
                __m256 g = _mm256_mul_ps(_mm256_sub_ps(on, pv), nv);
                s = _mm256_fmadd_ps(pv, s, g);
                _mm256_storeu_ps(ob + off, s);
            }
            _mm256_storeu_ps(send + (size_t)b * N + d, s);
        }
        for (; d < N; ++d) {
            float s = s0b[d];
            for (int t = 0; t < T; ++t) {
                size_t off = (size_t)t * N + d;
                float pv = pb[off];
                s = pv * s + (one - pv) * nb[off];
                ob[off] = s;
            }
            send[(size_t)b * N + d] = s;
        }
    }
}

/* 反向: 上游 dout (B,T,N), 累加 dp/dnu/ds0. out 必须保留 (偏导依赖前向状态).
 * ds = dout_t + p_{t+1} * ds_{t+1}
 */
EXPORT void scan_bwd_f32(const float* p, const float* nu, const float* s0,
                         const float* out, const float* dout,
                         float* dp, float* dnu, float* ds0,
                         int B, int T, int N) {
    const float one = 1.0f;
    for (int b = 0; b < B; ++b) {
        const float* pb = p + (size_t)b * T * N;
        const float* nb = nu + (size_t)b * T * N;
        const float* s0b = s0 + (size_t)b * N;
        const float* ob = out + (size_t)b * T * N;
        const float* db = dout + (size_t)b * T * N;
        float* dpb = dp + (size_t)b * T * N;
        float* dnb = dnu + (size_t)b * T * N;
        float* d0b = ds0 + (size_t)b * N;
        int d = 0;
        for (; d + 8 <= N; d += 8) {
            __m256 ds = _mm256_setzero_ps();
            const __m256 on = _mm256_set1_ps(one);
            for (int t = T - 1; t >= 0; --t) {
                size_t off = (size_t)t * N + d;
                __m256 do_ = _mm256_loadu_ps(db + off);
                ds = _mm256_add_ps(do_, ds);
                __m256 pv = _mm256_loadu_ps(pb + off);
                __m256 nv = _mm256_loadu_ps(nb + off);
                __m256 sprev = (t > 0)
                    ? _mm256_loadu_ps(ob + (size_t)(t - 1) * N + d)
                    : _mm256_loadu_ps(s0b + d);
                __m256 dnv = _mm256_mul_ps(ds, _mm256_sub_ps(on, pv));
                __m256 dpv = _mm256_mul_ps(ds, _mm256_sub_ps(sprev, nv));
                _mm256_storeu_ps(dnb + off, dnv);
                _mm256_storeu_ps(dpb + off, dpv);
                ds = _mm256_mul_ps(ds, pv);
            }
            _mm256_storeu_ps(d0b + d, ds);
        }
        for (; d < N; ++d) {
            float ds = 0.0f;
            for (int t = T - 1; t >= 0; --t) {
                size_t off = (size_t)t * N + d;
                ds += db[off];
                float pv = pb[off], nv = nb[off];
                float sprev = (t > 0) ? ob[(size_t)(t - 1) * N + d] : s0b[d];
                dnb[off] = ds * (one - pv);
                dpb[off] = ds * (sprev - nv);
                ds *= pv;
            }
            d0b[d] = ds;
        }
    }
}