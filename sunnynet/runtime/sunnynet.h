/* SunnyNet Runtime: whole-model decode step in C (no Python/PyTorch).
 *
 * Loads a packed fp32 weight blob exported from a checkpoint, keeps rolling
 * state + conv history per layer, and produces logits for one token.
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

#ifdef __cplusplus
}
#endif
#endif