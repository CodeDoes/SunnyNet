/* SunnyNet Runtime: whole-model decode step in C (no Python/PyTorch).
 *
 * Loads a packed fp32 weight blob exported from a checkpoint, keeps rolling
 * state + conv history per layer, and produces logits for one token.
 *
 * Weights live row-major (out, in) in memory, so each output row of a decode
 * GEMV is a contiguous read. The exporter writes them transposed (in, out);
 * sn_load flips them on the way in. Rows are independent, so the GEMV is
 * parallel over output channels.
 */
#ifndef SN_MODEL_H
#define SN_MODEL_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct SnModel SnModel;

/* path: packed fp32 weights. tied=1 shares embed for the LM head.
 * returns NULL on failure. */
SnModel* sn_load(const char* path, int vocab, int dim, int inner, int rank,
                 int taps, int layers, int group, int tied);

void sn_free(SnModel* m);
void sn_reset(SnModel* m);

/* one decode step: token -> logits[vocab]. */
void sn_step(SnModel* m, int token, float* logits);

/* feed n tokens in order; logits of the last token written to out. */
void sn_prefill(SnModel* m, const int* tokens, int n, float* logits);

int sn_vocab(const SnModel* m);
int sn_ready(const SnModel* m);

/* GEMV 并行度; n<=0 维持 OpenMP 默认 (全部逻辑核). */
void sn_set_threads(int n);
int sn_get_threads(void);

/* 物理核数 (解码访存受限, 用它比逻辑核更快); 探测失败返回 0. */
int sn_recommended_threads(void);

#ifdef __cplusplus
}
#endif
#endif