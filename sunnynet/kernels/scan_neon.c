/* SunnyNet 递推内核: ARM NEON (AArch64 / ARMv8-A) 版本.
 *
 * 与 x86 版数值等价。AArch64 的 NEON 是必备特性, 无需运行期探测;
 * 每寄存器 4 个 float, 用 vfmaq_f32 做融合乘加。
 *
 * 编译 (树莓派 / Apple Silicon / 任何 AArch64 Linux):
 *   gcc -O3 -fopenmp -shared -fPIC scan_neon.c -o scan_neon.so
 * 注意不要加 -mfpu=neon: 那是 ARM32 专有选项, AArch64 的 gcc 会直接拒绝。
 * 交叉编译 (x86 主机):
 *   aarch64-linux-gnu-gcc -O3 -fopenmp -shared -fPIC scan_neon.c -o scan_neon.so
 * 或 zig cc -target aarch64-linux-gnu ...
 */
#include <arm_neon.h>
#include <math.h>
#include <stddef.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define EXPORT __attribute__((visibility("default")))

/* vfmaq_f32 需要 NEONv2 (AArch64, 或 32 位 ARM 的 -mfpu=neon-vfpv4)。
 * 老 32 位 ARMv7 工具链没有它, 退回 mul+add; 数值一致, 少一次舍入。 */
#if defined(__aarch64__) || defined(__ARM_FEATURE_FMA)
#define SN_FMA(acc, x, y) vfmaq_f32((acc), (x), (y))
#else
#define SN_FMA(acc, x, y) vaddq_f32((acc), vmulq_f32((x), (y)))
#endif

EXPORT void sn_neon_set_threads(int n) {
#ifdef _OPENMP
    omp_set_num_threads(n);
#else
    (void)n;
#endif
}

EXPORT int sn_neon_threads(void) {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

EXPORT const char* sn_neon_isa(void) { return "neon"; }

/* 前向扫描: s_t = p_t*s_{t-1} + (1-p_t)*nu_t. */
EXPORT void scan_neon_f32(const float* p, const float* nu, const float* s0,
                          float* out, float* send, int B, int T, int N) {
    const float32x4_t one = vdupq_n_f32(1.0f);
    for (int b = 0; b < B; ++b) {
        const float* pb = p + (size_t)b * T * N;
        const float* nb = nu + (size_t)b * T * N;
        float* ob = out + (size_t)b * T * N;
        const float* s0b = s0 + (size_t)b * N;
        float* seb = send + (size_t)b * N;
        int d = 0;
        for (; d + 4 <= N; d += 4) {
            float32x4_t s = vld1q_f32(s0b + d);
            for (int t = 0; t < T; ++t) {
                const size_t off = (size_t)t * N + d;
                float32x4_t pv = vld1q_f32(pb + off);
                float32x4_t nv = vld1q_f32(nb + off);
                float32x4_t g = vmulq_f32(vsubq_f32(one, pv), nv);
                s = SN_FMA(g, pv, s);
                vst1q_f32(ob + off, s);
            }
            vst1q_f32(seb + d, s);
        }
        for (; d < N; ++d) {
            float s = s0b[d];
            for (int t = 0; t < T; ++t) {
                const size_t off = (size_t)t * N + d;
                float pv = pb[off];
                s = pv * s + (one[0] - pv) * nb[off];
                ob[off] = s;
            }
            seb[d] = s;
        }
    }
}

/* 反向. */
EXPORT void scan_neon_bwd_f32(const float* p, const float* nu, const float* s0,
                              const float* out, const float* dout,
                              float* dp, float* dnu, float* ds0,
                              int B, int T, int N) {
    const float32x4_t one = vdupq_n_f32(1.0f);
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
        for (; d + 4 <= N; d += 4) {
            float32x4_t ds = vdupq_n_f32(0.0f);
            for (int t = T - 1; t >= 0; --t) {
                const size_t off = (size_t)t * N + d;
                ds = vaddq_f32(ds, vld1q_f32(db + off));
                float32x4_t pv = vld1q_f32(pb + off);
                float32x4_t nv = vld1q_f32(nb + off);
                float32x4_t sprev = (t > 0)
                    ? vld1q_f32(ob + (size_t)(t - 1) * N + d)
                    : vld1q_f32(s0b + d);
                float32x4_t dnv = vmulq_f32(ds, vsubq_f32(one, pv));
                float32x4_t dpv = vmulq_f32(ds, vsubq_f32(sprev, nv));
                vst1q_f32(dnb + off, dnv);
                vst1q_f32(dpb + off, dpv);
                ds = vmulq_f32(ds, pv);
            }
            vst1q_f32(d0b + d, ds);
        }
        for (; d < N; ++d) {
            float ds = 0.0f;
            for (int t = T - 1; t >= 0; --t) {
                const size_t off = (size_t)t * N + d;
                ds += db[off];
                float pv = pb[off], nv = nb[off];
                float sprev = (t > 0) ? ob[(size_t)(t - 1) * N + d] : s0b[d];
                dnb[off] = ds * (one[0] - pv);
                dpb[off] = ds * (sprev - nv);
                ds *= pv;
            }
            d0b[d] = ds;
        }
    }
}

/* 单步. */
EXPORT void step_neon_f32(const float* a, const float* b, float* s,
                          int B, int N, int gated) {
    const float32x4_t one = vdupq_n_f32(1.0f);
    for (int bb = 0; bb < B; ++bb) {
        const float* ab = a + (size_t)bb * N;
        const float* bbb = b + (size_t)bb * N;
        float* sb = s + (size_t)bb * N;
        int d = 0;
        for (; d + 4 <= N; d += 4) {
            float32x4_t av = vld1q_f32(ab + d);
            float32x4_t bv = vld1q_f32(bbb + d);
            if (gated) bv = vmulq_f32(vsubq_f32(one, av), bv);
            float32x4_t sv = vld1q_f32(sb + d);
            vst1q_f32(sb + d, SN_FMA(bv, av, sv));
        }
        for (; d < N; ++d) {
            const float av = ab[d];
            const float bv = gated ? (one[0] - av) * bbb[d] : bbb[d];
            sb[d] = av * sb[d] + bv;
        }
    }
}

/* 推理态单步 core: 组内成对竞争 + 门控概率 + 一步状态更新.
 * 与 step_avx2.c 的 step_core_f32 数值等价。组内 8 通道的竞争关系需要邻元,
 * 故整组先读进局部数组, 状态更新仍走 NEON 的 4-lane 块。
 */
EXPORT void step_core_neon_f32(const float* q, const float* delta,
                               const float* s, float* s_new, float* u_out,
                               int B, int D) {
    const int NG = D >> 3;
    const float32x4_t one = vdupq_n_f32(1.0f);
    for (int b = 0; b < B; ++b) {
        const float* qb = q + (size_t)b * 2 * D;
        const float* sb = s + (size_t)b * D;
        float* sbn = s_new + (size_t)b * D;
        float* ub = u_out + (size_t)b * D;
        for (int g = 0; g < NG; ++g) {
            const int d0 = g << 3;
            float e8[8], ur8[8], p8[8], u8[8];
            for (int i = 0; i < 8; ++i) {
                ur8[i] = qb[d0 + i];
                e8[i] = expf(-qb[D + d0 + i]);
            }
            for (int i = 0; i < 8; ++i) {
                float sig = 1.0f / (1.0f + e8[i]);
                float s1_ = 1.0f / (1.0f + e8[i] / e8[(i + 1) & 7]);
                float s2_ = 1.0f / (1.0f + e8[i] / e8[(i + 2) & 7]);
                float dli = delta[d0 + i];
                u8[i] = ur8[i] * s1_ * s2_;
                p8[i] = dli * sig + (1.0f - dli);
            }
            float32x4_t pl = vld1q_f32(p8);
            float32x4_t ul = vld1q_f32(u8);
            float32x4_t sl = vld1q_f32(sb + d0);
            sl = SN_FMA(vmulq_f32(vsubq_f32(one, pl), ul), pl, sl);
            vst1q_f32(sbn + d0, sl);
            vst1q_f32(ub + d0, ul);

            float32x4_t ph = vld1q_f32(p8 + 4);
            float32x4_t uh = vld1q_f32(u8 + 4);
            float32x4_t sh = vld1q_f32(sb + d0 + 4);
            sh = SN_FMA(vmulq_f32(vsubq_f32(one, ph), uh), ph, sh);
            vst1q_f32(sbn + d0 + 4, sh);
            vst1q_f32(ub + d0 + 4, uh);
        }
    }
}

/* core: 组内成对竞争门控 + 门控概率 + 扫描, 单趟. group 恒 8 (两块 NEON).
 * 与 core_avx2.c 的 core_fwd_f32 数值等价。 */
EXPORT void core_neon_fwd_f32(const float* q, const float* delta,
                              const float* s0, float* states, float* uo,
                              float* po, int B, int T, int D) {
    const int NG = D >> 3;
    const float32x4_t one = vdupq_n_f32(1.0f);
    for (int b = 0; b < B; ++b) {
        for (int g = 0; g < NG; ++g) {
            const int d0 = g << 3;
            const size_t qb = (size_t)b * T * 2 * D;
            const size_t sb = (size_t)b * T * D;
            /* 组内 8 个通道切成两个 4-lane 块, 成对竞争需跨块访问邻元,
             * 故先整组读到局部数组, 再用 NEON 处理。 */
            float gt8[8];
            float32x4_t s01 = vld1q_f32(s0 + (size_t)b * D + d0);
            float32x4_t s02 = vld1q_f32(s0 + (size_t)b * D + d0 + 4);
            float dl1[4], dl2[4];
            vst1q_f32(dl1, vld1q_f32(delta + d0));
            vst1q_f32(dl2, vld1q_f32(delta + d0 + 4));
            for (int t = 0; t < T; ++t) {
                const size_t qoff = qb + (size_t)t * 2 * D + d0;
                const size_t off = sb + (size_t)t * D + d0;
                float ur8[8], g8[8];
                for (int i = 0; i < 8; ++i) {
                    ur8[i] = q[qoff + i];
                    g8[i] = q[qoff + D + i];
                    gt8[i] = expf(-g8[i]);
                }
                float u8[8], p8[8];
                for (int i = 0; i < 8; ++i) {
                    float sig = 1.0f / (1.0f + gt8[i]);
                    float s1_ = 1.0f / (1.0f + gt8[i] / gt8[(i + 1) & 7]);
                    float s2_ = 1.0f / (1.0f + gt8[i] / gt8[(i + 2) & 7]);
                    float w = s1_ * s2_;
                    float dli = (i < 4) ? dl1[i] : dl2[i - 4];
                    u8[i] = ur8[i] * w;
                    p8[i] = dli * sig + (1.0f - dli);
                }
                float32x4_t pv1 = vld1q_f32(p8);
                float32x4_t uv1 = vld1q_f32(u8);
                s01 = SN_FMA(vmulq_f32(vsubq_f32(one, pv1), uv1), pv1, s01);
                vst1q_f32(states + off, s01);
                vst1q_f32(uo + off, uv1);
                vst1q_f32(po + off, pv1);

                float32x4_t pv2 = vld1q_f32(p8 + 4);
                float32x4_t uv2 = vld1q_f32(u8 + 4);
                s02 = SN_FMA(vmulq_f32(vsubq_f32(one, pv2), uv2), pv2, s02);
                vst1q_f32(states + off + 4, s02);
                vst1q_f32(uo + off + 4, uv2);
                vst1q_f32(po + off + 4, pv2);
            }
        }
    }
}