/* SunnyNet Runtime: Mamba-2 单步解码, 整模型在 C (AVX2/FMA).
 *
 * 每 head 状态在数学上是 (head_dim, state_size), 但衰减 dA = exp(dt*A)
 * 对 state 维全同、外积可分解, 故退化为元素级一阶递推:
 *   S_t = S_{t-1} * dA + (dt * B) (x) x_ssm
 * 权重从打包 fp32 blob 读入 (由 experiments/_export_mamba2.py 生成).
 */
#ifndef SN_MAMBA2_H
#define SN_MAMBA2_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct SnMamba2 SnMamba2;

SnMamba2* sn_m2_load(const char* path);
void sn_m2_free(SnMamba2* m);
void sn_m2_reset(SnMamba2* m);

/* 单步解码访存受限, 线程数取物理核数 (取 0 或负数即不限制)。 */
void sn_m2_set_threads(int n);
int sn_m2_get_threads(void);

/* token -> logits[vocab]. */
void sn_m2_step(SnMamba2* m, int token, float* logits);
void sn_m2_prefill(SnMamba2* m, const int* tokens, int n, float* logits);

/* 跑完第 layer 层后的主流 x (D 个 float); layer<0 跑完全部层。逐层对表用。 */
void sn_m2_dbg(SnMamba2* m, int token, int layer, float* out);

int sn_m2_vocab(const SnMamba2* m);
int sn_m2_ready(const SnMamba2* m);

#ifdef __cplusplus
}
#endif
#endif