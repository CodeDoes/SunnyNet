/* SunnyNet 融合 core 内核: 组内成对竞争门控 + 门控概率 + 扫描, 单趟流过.
 * group 恒为 8 (SunnyBlock._pick_group 在本项目各预设下都返回 8), 正好一个 AVX2
 * 寄存器, torch.roll 退化为 __m256_permutevar8x32_ps 的 lane 置换.
 * sigmoid(g_i - g_j) 用 e_j/e_i 比值: e = exp(-g) 每 lane 一次, 替代三次 exp.
 */
#include <immintrin.h>
#include <math.h>
#include <stddef.h>
#include <omp.h>

#define EXPORT __declspec(dllexport)

static const int IDX_L1[8] = {1, 2, 3, 4, 5, 6, 7, 0};  /* v[i+1] */
static const int IDX_L2[8] = {2, 3, 4, 5, 6, 7, 0, 1};  /* v[i+2] */
static const int IDX_R1[8] = {7, 0, 1, 2, 3, 4, 5, 6};  /* v[i-1] */
static const int IDX_R2[8] = {6, 7, 0, 1, 2, 3, 4, 5};  /* v[i-2] */

EXPORT void sn_set_threads(int n) { omp_set_num_threads(n); }

static inline __m256 expv(__m256 x) {
    float a[8];
    _mm256_storeu_ps(a, x);
    for (int i = 0; i < 8; ++i) a[i] = expf(a[i]);
    return _mm256_loadu_ps(a);
}

/* q: (B,T,2D) 连续, [0,D) = u_raw, [D,2D) = gate.  delta: (D,).  s0: (B,D).
 * 出: states (B,T,D), u (B,T,D), p (B,T,D).
 */
EXPORT void core_fwd_f32(const float* q, const float* delta, const float* s0,
                         float* states, float* uo, float* po,
                         int B, int T, int D) {
    const int NG = D >> 3;
    const __m256i il1 = _mm256_loadu_si256((const __m256i*)IDX_L1);
    const __m256i il2 = _mm256_loadu_si256((const __m256i*)IDX_L2);
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 zero = _mm256_setzero_ps();

    #pragma omp parallel for collapse(2) schedule(static)
    for (int b = 0; b < B; ++b) {
        for (int g = 0; g < NG; ++g) {
            const int d0 = g << 3;
            const size_t qb = (size_t)b * T * 2 * D;
            const size_t sb = (size_t)b * T * D;
            __m256 s = _mm256_loadu_ps(s0 + (size_t)b * D + d0);
            __m256 dl = _mm256_loadu_ps(delta + d0);
            __m256 base = _mm256_sub_ps(one, dl);
            for (int t = 0; t < T; ++t) {
                const size_t qoff = qb + (size_t)t * 2 * D + d0;
                const size_t off = sb + (size_t)t * D + d0;
                __m256 ur = _mm256_loadu_ps(q + qoff);
                __m256 gt = _mm256_loadu_ps(q + qoff + D);
                __m256 e = expv(_mm256_sub_ps(zero, gt));
                __m256 e1 = _mm256_permutevar8x32_ps(e, il1);
                __m256 e2 = _mm256_permutevar8x32_ps(e, il2);
                __m256 sig = _mm256_div_ps(one, _mm256_add_ps(one, e));
                __m256 s1 = _mm256_div_ps(one, _mm256_add_ps(one, _mm256_div_ps(e, e1)));
                __m256 s2 = _mm256_div_ps(one, _mm256_add_ps(one, _mm256_div_ps(e, e2)));
                __m256 w = _mm256_mul_ps(s1, s2);
                __m256 u = _mm256_mul_ps(ur, w);
                __m256 p = _mm256_fmadd_ps(dl, sig, base);
                s = _mm256_fmadd_ps(p, s, _mm256_mul_ps(_mm256_sub_ps(one, p), u));
                _mm256_storeu_ps(states + off, s);
                _mm256_storeu_ps(uo + off, u);
                _mm256_storeu_ps(po + off, p);
            }
        }
    }
}

/* 反向: 上游 dstate (B,T,D, 扫描支路) 与 du_in (B,T,D, _emit 支路).
 * 出 dq (B,T,2D), ddelta (D,), ds0 (B,D).
 */
EXPORT void core_bwd_f32(const float* q, const float* delta, const float* s0,
                         const float* states, const float* uo, const float* po,
                         const float* dstate, const float* du_in,
                         float* dq, float* ddelta, float* ds0,
                         int B, int T, int D) {
    const int NG = D >> 3;
    const __m256i il1 = _mm256_loadu_si256((const __m256i*)IDX_L1);
    const __m256i il2 = _mm256_loadu_si256((const __m256i*)IDX_L2);
    const __m256i ir1 = _mm256_loadu_si256((const __m256i*)IDX_R1);
    const __m256i ir2 = _mm256_loadu_si256((const __m256i*)IDX_R2);
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 zero = _mm256_setzero_ps();

    #pragma omp parallel for collapse(2) schedule(static)
    for (int b = 0; b < B; ++b) {
        for (int g = 0; g < NG; ++g) {
            const int d0 = g << 3;
            const size_t qb = (size_t)b * T * 2 * D;
            const size_t sb = (size_t)b * T * D;
            const float* s0b = s0 + (size_t)b * D + d0;
            __m256 dl = _mm256_loadu_ps(delta + d0);
            __m256 dacc = _mm256_setzero_ps();
            __m256 ds = _mm256_setzero_ps();
            for (int t = T - 1; t >= 0; --t) {
                const size_t qoff = qb + (size_t)t * 2 * D + d0;
                const size_t off = sb + (size_t)t * D + d0;
                ds = _mm256_add_ps(ds, _mm256_loadu_ps(dstate + off));
                __m256 u = _mm256_loadu_ps(uo + off);
                __m256 p = _mm256_loadu_ps(po + off);
                __m256 sprev = (t > 0) ? _mm256_loadu_ps(states + off - D)
                                       : _mm256_loadu_ps(s0b);
                __m256 dp = _mm256_mul_ps(ds, _mm256_sub_ps(sprev, u));
                __m256 du = _mm256_fmadd_ps(ds, _mm256_sub_ps(one, p),
                                            _mm256_loadu_ps(du_in + off));
                ds = _mm256_mul_ps(ds, p);

                __m256 ur = _mm256_loadu_ps(q + qoff);
                __m256 gt = _mm256_loadu_ps(q + qoff + D);
                __m256 e = expv(_mm256_sub_ps(zero, gt));
                __m256 e1 = _mm256_permutevar8x32_ps(e, il1);
                __m256 e2 = _mm256_permutevar8x32_ps(e, il2);
                __m256 sig = _mm256_div_ps(one, _mm256_add_ps(one, e));
                __m256 s1 = _mm256_div_ps(one, _mm256_add_ps(one, _mm256_div_ps(e, e1)));
                __m256 s2 = _mm256_div_ps(one, _mm256_add_ps(one, _mm256_div_ps(e, e2)));
                __m256 w = _mm256_mul_ps(s1, s2);

                __m256 dur = _mm256_mul_ps(du, w);
                __m256 dw = _mm256_mul_ps(du, ur);
                __m256 ds1 = _mm256_mul_ps(dw, s2);
                __m256 ds2 = _mm256_mul_ps(dw, s1);
                __m256 a1 = _mm256_mul_ps(ds1, _mm256_mul_ps(s1, _mm256_sub_ps(one, s1)));
                __m256 a2 = _mm256_mul_ps(ds2, _mm256_mul_ps(s2, _mm256_sub_ps(one, s2)));
                __m256 dg = _mm256_add_ps(a1, a2);
                dg = _mm256_sub_ps(dg, _mm256_permutevar8x32_ps(a1, ir1));
                dg = _mm256_sub_ps(dg, _mm256_permutevar8x32_ps(a2, ir2));
                dg = _mm256_fmadd_ps(_mm256_mul_ps(dp, dl),
                                     _mm256_mul_ps(sig, _mm256_sub_ps(one, sig)), dg);
                dacc = _mm256_fmadd_ps(dp, _mm256_sub_ps(sig, one), dacc);

                _mm256_storeu_ps(dq + qoff, dur);
                _mm256_storeu_ps(dq + qoff + D, dg);
            }
            _mm256_storeu_ps(ds0 + (size_t)b * D + d0, ds);
            float tmp[8];
            _mm256_storeu_ps(tmp, dacc);
            for (int i = 0; i < 8; ++i) {
                #pragma omp atomic
                ddelta[d0 + i] += tmp[i];
            }
        }
    }
}