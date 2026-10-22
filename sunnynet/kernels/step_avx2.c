/* SunnyNet 推理态单步内核. 解码每步形状 (B,1,D), torch 侧要发起十几条 aten 调用. */
#include <immintrin.h>
#include <math.h>
#include <stddef.h>
#include <omp.h>

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

static const int IDX_L1[8] = {1, 2, 3, 4, 5, 6, 7, 0};
static const int IDX_L2[8] = {2, 3, 4, 5, 6, 7, 0, 1};

static inline __m256 expv8(__m256 x) {
    float a[8];
    _mm256_storeu_ps(a, x);
    for (int i = 0; i < 8; ++i) a[i] = expf(a[i]);
    return _mm256_loadu_ps(a);
}

/* w 布局 (taps, D); hist[:,0] 是上一时刻输入. out[d] = w[taps-1][d]*x[d] + sum_j w[taps-2-j][d]*hist[j][d] */
EXPORT void dw_step_f32(const float* x, const float* hist, const float* w,
                        float* acc, int B, int D, int taps) {
    const int k = taps - 1;
    #pragma omp parallel for schedule(static) if(B * (D >> 3) >= 256)
    for (int b = 0; b < B; ++b) {
        const float* xb = x + (size_t)b * D;
        const float* hb = hist + (size_t)b * k * D;
        float* ab = acc + (size_t)b * D;
        const float* wt = w + (size_t)(taps - 1) * D;
        int d = 0;
        for (; d + 8 <= D; d += 8) {
            __m256 a = _mm256_mul_ps(_mm256_loadu_ps(xb + d),
                                     _mm256_loadu_ps(wt + d));
            for (int j = 0; j < k; ++j) {
                __m256 hv = _mm256_loadu_ps(hb + (size_t)j * D + d);
                __m256 wv = _mm256_loadu_ps(w + (size_t)(taps - 2 - j) * D + d);
                a = _mm256_fmadd_ps(hv, wv, a);
            }
            _mm256_storeu_ps(ab + d, a);
        }
        for (; d < D; ++d) {
            float a = xb[d] * wt[d];
            for (int j = 0; j < k; ++j)
                a += hb[(size_t)j * D + d] * w[(size_t)(taps - 2 - j) * D + d];
            ab[d] = a;
        }
    }
}

/* q (B,2D) 的 [0,D)=u_raw, [D,2D)=gate; s (B,D) -> s_new, u_out; delta (D,). */
EXPORT void step_core_f32(const float* q, const float* delta, const float* s,
                          float* s_new, float* u_out, int B, int D) {
    const int NG = D >> 3;
    const __m256i il1 = _mm256_loadu_si256((const __m256i*)IDX_L1);
    const __m256i il2 = _mm256_loadu_si256((const __m256i*)IDX_L2);
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 zero = _mm256_setzero_ps();
    const int work = B * NG;
    #pragma omp parallel for schedule(static) if(work >= 256)
    for (int b = 0; b < B; ++b) {
        const float* qb = q + (size_t)b * 2 * D;
        const float* sb = s + (size_t)b * D;
        float* sbn = s_new + (size_t)b * D;
        float* ub = u_out + (size_t)b * D;
        for (int g = 0; g < NG; ++g) {
            const int d0 = g << 3;
            __m256 ur = _mm256_loadu_ps(qb + d0);
            __m256 gt = _mm256_loadu_ps(qb + D + d0);
            __m256 dl = _mm256_loadu_ps(delta + d0);
            __m256 sv = _mm256_loadu_ps(sb + d0);
            __m256 e = expv8(_mm256_sub_ps(zero, gt));
            __m256 e1 = _mm256_permutevar8x32_ps(e, il1);
            __m256 e2 = _mm256_permutevar8x32_ps(e, il2);
            __m256 sig = _mm256_div_ps(one, _mm256_add_ps(one, e));
            __m256 s1 = _mm256_div_ps(one, _mm256_add_ps(one, _mm256_div_ps(e, e1)));
            __m256 s2 = _mm256_div_ps(one, _mm256_add_ps(one, _mm256_div_ps(e, e2)));
            __m256 u = _mm256_mul_ps(ur, _mm256_mul_ps(s1, s2));
            __m256 p = _mm256_fmadd_ps(dl, sig, _mm256_sub_ps(one, dl));
            __m256 sn = _mm256_fmadd_ps(p, sv, _mm256_mul_ps(_mm256_sub_ps(one, p), u));
            _mm256_storeu_ps(ub + d0, u);
            _mm256_storeu_ps(sbn + d0, sn);
        }
    }
}