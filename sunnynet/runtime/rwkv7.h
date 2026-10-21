/* SunnyNet Runtime: RWKV-7 单步解码, 整模型在 C (AVX2/FMA).
 *
 * 状态是逐 head 矩阵 S[h] = (HS,HS), 更新含 rank-1 校正:
 *   S_t = S_{t-1} * w - (S_{t-1} kk) x (kk * a) + v x k
 * 权重从打包 fp32 blob 读入 (由 experiments/_export_rwkv7.py 从 GGUF 生成).
 */
#ifndef SN_RWKV7_H
#define SN_RWKV7_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct SnRWKV7 SnRWKV7;

SnRWKV7* sn_rk_load(const char* path);
void sn_rk_free(SnRWKV7* m);
void sn_rk_reset(SnRWKV7* m);

/* token -> logits[vocab]. */
void sn_rk_step(SnRWKV7* m, int token, float* logits);
void sn_rk_prefill(SnRWKV7* m, const int* tokens, int n, float* logits);

/* 逐层中间量转储: xn, r, k, v, a, g, ww, y, x_out 各 (D,). */
void sn_rk_dbg(SnRWKV7* m, int token, int layer, float* out);

int sn_rk_vocab(const SnRWKV7* m);
int sn_rk_ready(const SnRWKV7* m);

#ifdef __cplusplus
}
#endif
#endif