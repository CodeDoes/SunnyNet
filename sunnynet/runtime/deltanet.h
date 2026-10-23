/* SunnyNet Runtime: DeltaNet 单步解码, 整模型在 C (AVX2/FMA).
 *
 * 状态是逐 head 的 (K,V) 矩阵, delta rule 含 rank-1 校正:
 *   delta_t = (v_t - S_{t-1}^T k_t) * beta_t
 *   S_t     = S_{t-1} + k_t x delta_t
 *   o_t     = S_t^T q_t
 * 权重从打包 fp32 blob 读入 (由 experiments/_export_deltanet.py 生成).
 * 本仓把 S 存成 (k_dim, v_dim) 行主序, 与 "S^T k" 的写法对应。
 */
#ifndef SN_DELTANET_H
#define SN_DELTANET_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct SnDeltaNet SnDeltaNet;

SnDeltaNet* sn_dn_load(const char* path);
void sn_dn_free(SnDeltaNet* m);
void sn_dn_reset(SnDeltaNet* m);

/* 单步解码访存受限, 线程数取物理核数 (取 0 或负数即不限制)。 */
void sn_dn_set_threads(int n);
int sn_dn_get_threads(void);

/* token -> logits[vocab]. */
void sn_dn_step(SnDeltaNet* m, int token, float* logits);
void sn_dn_prefill(SnDeltaNet* m, const int* tokens, int n, float* logits);

/* 跑完第 layer 层后的主流 x (D 个 float); layer<0 跑完全部层。逐层对表用。 */
void sn_dn_dbg(SnDeltaNet* m, int token, int layer, float* out);

int sn_dn_vocab(const SnDeltaNet* m);
int sn_dn_ready(const SnDeltaNet* m);

#ifdef __cplusplus
}
#endif
#endif