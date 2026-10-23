/* SunnyNet Runtime: HGRN2 单步解码, 整模型在 C (AVX2/FMA).
 *
 * 状态是逐 head 的 (K,V) 矩阵, 衰减只作用在 key 维:
 *   q = silu(q_proj(h))
 *   f = f_proj(h);  g = logaddexp(lb, log1p(-lb) + logsigmoid(f))
 *   k = 1 - exp(g)                     写权重由衰减导出, 没有 k_proj
 *   S_t = diag(exp(g)) S_{t-1} + k_t (x) i_t
 *   o_t = scale * S_t^T q_t
 * 无输出门: o 在 value 维整体 RMSNorm 后接 o_proj。
 * 权重从打包 fp32 blob 读入 (由 experiments/_export_hgrn2.py 生成).
 */
#ifndef SN_HGRN2_H
#define SN_HGRN2_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct SnHGRN2 SnHGRN2;

SnHGRN2* sn_hg_load(const char* path);
void sn_hg_free(SnHGRN2* m);
void sn_hg_reset(SnHGRN2* m);

/* 单步解码访存受限, 线程数取物理核数。 */
void sn_hg_set_threads(int n);
int sn_hg_get_threads(void);

/* token -> logits[vocab]. */
void sn_hg_step(SnHGRN2* m, int token, float* logits);
void sn_hg_prefill(SnHGRN2* m, const int* tokens, int n, float* logits);

/* 跑完第 layer 层后的主流 x (D 个 float); layer<0 跑完全部层。 */
void sn_hg_dbg(SnHGRN2* m, int token, int layer, float* out);

/* 层内阶段探针。stage: 0 attn_norm | 1 q(silu 后) | 2 f | 3 g | 4 exp(g)
 *   5 k=1-exp(g) | 6 i(value) | 7 读出 | 8 整体 RMSNorm 后 | 9 o_proj */
void sn_hg_probe(SnHGRN2* m, int token, int layer, int stage, float* out);

int sn_hg_vocab(const SnHGRN2* m);
int sn_hg_ready(const SnHGRN2* m);

#ifdef __cplusplus
}
#endif
#endif