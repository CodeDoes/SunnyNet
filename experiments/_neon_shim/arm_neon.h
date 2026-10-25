/* 在 x86 上跑真 NEON 源码用的标量 shim.
 * 目的不是提速, 而是让 scan_neon.c 的索引与组内语义在能被执行的平台上跑一遍:
 * 把 4-lane 向量当成 4 元 float 数组, 运算逐元素做, 与 NEON 语义一致。 */
#ifndef SN_NEON_SHIM_H
#define SN_NEON_SHIM_H

typedef float float32x4_t __attribute__((vector_size(16)));

static inline float32x4_t vdupq_n_f32(float x) {
    return (float32x4_t){x, x, x, x};
}
static inline float32x4_t vld1q_f32(const float* p) {
    return (float32x4_t){p[0], p[1], p[2], p[3]};
}
static inline void vst1q_f32(float* p, float32x4_t v) {
    p[0] = v[0]; p[1] = v[1]; p[2] = v[2]; p[3] = v[3];
}
static inline float32x4_t vaddq_f32(float32x4_t a, float32x4_t b) { return a + b; }
static inline float32x4_t vsubq_f32(float32x4_t a, float32x4_t b) { return a - b; }
static inline float32x4_t vmulq_f32(float32x4_t a, float32x4_t b) { return a * b; }
static inline float32x4_t vfmaq_f32(float32x4_t a, float32x4_t b, float32x4_t c) {
    return a + b * c;
}
#endif
