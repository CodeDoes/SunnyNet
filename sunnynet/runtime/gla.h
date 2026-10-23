/* SunnyNet Runtime: GLA 单步解码, 整模型在 C (AVX2/FMA).
 *
 * 状态是逐 head 的 (K,V) 矩阵, 但衰减 gk 只作用在 key 维 (low-rank 投影 16 维),
 * 故递推可写成按行缩放, 不需要矩阵值层级:
 *   S_t = diag(exp(gk_t)) S_{t-1} + k_t (x) v_t
 *   o_t = scale * S_t^T q_t
 * 输出门: y = RMSNorm(o) * g_norm * silu(g_proj(h)), 逐 head 在 head_v_dim 上归一。
 * 权重从打包 fp32 blob 读入 (由 experiments/_export_gla.py 生成).
 */
#ifndef SN_GLA_H
#define SN_GLA_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct SnGLA SnGLA;

SnGLA* sn_gla_load(const char* path);
void sn_gla_free(SnGLA* m);
void sn_gla_reset(SnGLA* m);

/* 单步解码访存受限, 线程数取物理核数 (取 0 或负数即不限制)。 */
void sn_gla_set_threads(int n);
int sn_gla_get_threads(void);

/* token -> logits[vocab]. */
void sn_gla_step(SnGLA* m, int token, float* logits);
void sn_gla_prefill(SnGLA* m, const int* tokens, int n, float* logits);

/* 跑完第 layer 层后的主流 x (D 个 float); layer<0 跑完全部层。逐层对表用。 */
void sn_gla_dbg(SnGLA* m, int token, int layer, float* out);

/* 层内阶段探针, 供定位用。stage 编号见 gla.c 的实现注释。 */
void sn_gla_probe(SnGLA* m, int token, int layer, int stage, float* out);

int sn_gla_vocab(const SnGLA* m);
int sn_gla_ready(const SnGLA* m);

#ifdef __cplusplus
}
#endif
#endif