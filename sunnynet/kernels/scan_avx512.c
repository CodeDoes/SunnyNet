/* SunnyNet 递推内核: AVX-512F/BW/VL 版本.
 *
 * 与 AVX2 版数值等价, 只是每寄存器 16 个 float。用掩码处理尾部, 故对任意 N
 * 都能走向量路径, 不再需要标量收尾循环。
 *
 * 编译: gcc -O3 -mavx512f -mavx512bw -mavx512vl -mfma -fopenmp -shared
 * 仅能在支持 AVX-512 的 CPU 上执行; 运行前须用 CPUID 探测 AVX512F。
 */
#include <immintrin.h>
#include <math.h>
#include <stddef.h>
#include <omp.h>

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

EXPORT void sn512_set_threads(int n) { omp_set_num_threads(n); }
EXPORT int sn512_threads(void) { return omp_get_max_threads(); }
EXPORT const char* sn512_isa(void) { return "avx512f+bw+vl+fma"; }

/* 前向扫描: s_t = p_t*s_{t-1} + (1-p_t)*nu_t.
 * p, nu: (B,T,N); s0, send: (B,N); out: (B,T,N). */
EXPORT void scan512_f32(const float* p, const float* nu, const float* s0,
                        float* out, float* send, int B, int T, int N) {
    const __m512 one = _mm512_set1_ps(1.0f);
    for (int b = 0; b < B; ++b) {
        const float* pb = p + (size_t)b * T * N;
        const float* nb = nu + (size_t)b * T * N;
        float* ob = out + (size_t)b * T * N;
        const float* s0b = s0 + (size_t)b * N;
        float* seb = send + (size_t)b * N;
        int d = 0;
        for (; d < N; d += 16) {
            const __mmask16 m = (N - d >= 16)
                ? (__mmask16)0xFFFF : (__mmask16)((1u << (N - d)) - 1);
            __m512 s = _mm512_maskz_loadu_ps(m, s0b + d);
            for (int t = 0; t < T; ++t) {
                const size_t off = (size_t)t * N + d;
                __m512 pv = _mm512_maskz_loadu_ps(m, pb + off);
                __m512 nv = _mm512_maskz_loadu_ps(m, nb + off);
                __m512 g = _mm512_mul_ps(_mm512_sub_ps(one, pv), nv);
                s = _mm512_fmadd_ps(pv, s, g);
                _mm512_mask_storeu_ps(ob + off, m, s);
            }
            _mm512_mask_storeu_ps(seb + d, m, s);
        }
    }
}

/* 反向, 语义同 scan_bwd_f32: ds = dout_t + p_{t+1}*ds_{t+1}. */
EXPORT void scan512_bwd_f32(const float* p, const float* nu, const float* s0,
                            const float* out, const float* dout,
                            float* dp, float* dnu, float* ds0,
                            int B, int T, int N) {
    const __m512 one = _mm512_set1_ps(1.0f);
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
        for (; d < N; d += 16) {
            const __mmask16 m = (N - d >= 16)
                ? (__mmask16)0xFFFF : (__mmask16)((1u << (N - d)) - 1);
            __m512 ds = _mm512_setzero_ps();
            for (int t = T - 1; t >= 0; --t) {
                const size_t off = (size_t)t * N + d;
                ds = _mm512_add_ps(ds, _mm512_maskz_loadu_ps(m, db + off));
                __m512 pv = _mm512_maskz_loadu_ps(m, pb + off);
                __m512 nv = _mm512_maskz_loadu_ps(m, nb + off);
                __m512 sprev = (t > 0)
                    ? _mm512_maskz_loadu_ps(m, ob + (size_t)(t - 1) * N + d)
                    : _mm512_maskz_loadu_ps(m, s0b + d);
                __m512 dnv = _mm512_mul_ps(ds, _mm512_sub_ps(one, pv));
                __m512 dpv = _mm512_mul_ps(ds, _mm512_sub_ps(sprev, nv));
                _mm512_mask_storeu_ps(dnb + off, m, dnv);
                _mm512_mask_storeu_ps(dpb + off, m, dpv);
                ds = _mm512_mul_ps(ds, pv);
            }
            _mm512_mask_storeu_ps(d0b + d, m, ds);
        }
    }
}

/* 单步递推: a, b, s 各 (B,N); s 原地更新. gated=1 时 b 视为 u. */
EXPORT void step512_f32(const float* a, const float* b, float* s,
                        int B, int N, int gated) {
    const __m512 one = _mm512_set1_ps(1.0f);
    for (int bb = 0; bb < B; ++bb) {
        const float* ab = a + (size_t)bb * N;
        const float* bbb = b + (size_t)bb * N;
        float* sb = s + (size_t)bb * N;
        int d = 0;
        for (; d < N; d += 16) {
            const __mmask16 m = (N - d >= 16)
                ? (__mmask16)0xFFFF : (__mmask16)((1u << (N - d)) - 1);
            __m512 av = _mm512_maskz_loadu_ps(m, ab + d);
            __m512 bv = _mm512_maskz_loadu_ps(m, bbb + d);
            if (gated) bv = _mm512_mul_ps(_mm512_sub_ps(one, av), bv);
            __m512 sv = _mm512_maskz_loadu_ps(m, sb + d);
            _mm512_mask_storeu_ps(sb + d, m, _mm512_fmadd_ps(av, sv, bv));
        }
    }
}

/* 运行期探测: 本库需 CPU 支持 AVX512F, 否则调用会触发非法指令.
 * 返回 1 表示可执行本库. 用 CPUID leaf 7 subleaf 0 的 EBX bit16。 */
EXPORT int sn512_cpu_supported(void) {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
    unsigned int eax, ebx, ecx, edx;
#if defined(_MSC_VER)
    int regs[4];
    __cpuid(regs, 7);
    ebx = (unsigned)regs[1];
    (void)eax; (void)ecx; (void)edx;
#else
    __asm__ volatile("cpuid"
                     : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx)
                     : "a"(7), "c"(0));
#endif
    /* 先确认 leaf 7 可用 */
    unsigned int maxleaf;
#if defined(_MSC_VER)
    int r0[4]; __cpuid(r0, 0); maxleaf = (unsigned)r0[0];
#else
    __asm__ volatile("cpuid" : "=a"(maxleaf) : "a"(0) : "ebx", "ecx", "edx");
#endif
    if (maxleaf < 7) return 0;
    return (ebx >> 16) & 1;
#else
    return 0;
#endif
}