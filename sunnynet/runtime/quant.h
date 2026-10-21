/* SunnyNet Runtime: 分组对称 INT4 权重量化 + 量化 GEMV (AVX2/FMA).
 *
 * 布局: W 存 (out, in) 行主序; 每 block 个连续 in 值共享一个 fp32 scale.
 * 打包: 组内字节 k 低半字节 = 第 k 个值, 高半字节 = 第 (k + block/2) 个值,
 * 值域 [-8, 7] 四比特补码. 要求 block 为 64 且整除 in.
 *
 * 量化 GEMV 读 4 bit/权重, 对内存带宽受限的解码是主要收益来源.
 */
#ifndef SN_QUANT_H
#define SN_QUANT_H

#ifdef __cplusplus
extern "C" {
#endif

/* 默认分组大小; 量化路径要求 in % SN_I4_BLOCK == 0. */
#define SN_I4_BLOCK 64

/* rows 行, 每行 N 列. q: (rows, N/2) 字节; scale: (rows, N/SN_I4_BLOCK). */
void sn_quant_i4(const float* x, unsigned char* q, float* scale,
                 int rows, int N, int block);
void sn_dequant_i4(const unsigned char* q, const float* scale, float* x,
                   int rows, int N, int block);

/* y = Wq x. Wq 存 (out, in) 量化; scale (out, in/block); x fp32. */
void sn_gemv_i4(float* y, const float* x, const unsigned char* Wq,
                const float* scale, int in, int out, int block);

/* fp32 参照, W 存 (out, in). */
void sn_gemv_f32(float* y, const float* x, const float* W, int in, int out);

#ifdef __cplusplus
}
#endif
#endif