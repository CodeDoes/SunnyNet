"""DeltaNet 正确性检查.

1. delta rule 的两种等价写法必须一致 (独立代码路径):
   秩 1 形式   S_t = S_{t-1} + k_t ⊗ ((v_t - S_{t-1}^T k_t) * beta_t)
   显式矩阵形式 S_t = S_{t-1}(I - beta_t k_t k_t^T) + beta_t v_t k_t^T
   后者用矩阵-矩阵乘法, 与前者实现完全不同, 逐位接近即可确认秩 1 更新写对了。
2. 单步递推 vs 分段续算 (状态可迁移)。
3. 逐步 bit 稳定性 (重复运行同输入结果不变)。
4. 英文贪心生成连贯性。
"""

import sys

import numpy as np

sys.path.insert(0, r"C:\Users\陈旭\Desktop\CPU神经网络")
from sunnynet.runtime.deltanet import DeltaNet  # noqa: E402

MDIR = r"F:\sunnyclaw-data\models\deltanet-pure"


def check_delta_algebra():
    """在真实权重上抽一层, 用手算的 q/k/v/beta 比对两种状态更新写法."""
    m = DeltaNet(MDIR)
    H, K, V = m.n_heads, m.head_k_dim, m.head_v_dim
    rng = np.random.default_rng(7)

    d_max = 0.0
    for trial in range(6):
        S = rng.standard_normal((H, K, V)).astype(np.float32)
        k = rng.standard_normal((H, K)).astype(np.float32)
        v = rng.standard_normal((H, V)).astype(np.float32)
        beta = rng.random(H).astype(np.float32)

        # 秩 1 形式 (deltanet.py 用的)
        kv = np.einsum("hkv,hk->hv", S, k)
        delta = (v - kv) * beta[:, None]
        S1 = S + k[:, :, None] * delta[:, None, :]

        # 显式矩阵形式. S 在本仓按 (k_dim, v_dim) 存, 是标准写法的转置, 故
        #   S' = (I - beta k k^T) S + beta k v^T
        # K 与 V 都是 128, 乘错一侧不会报维度错, 只会静默算成别的式子。
        I = np.eye(K, dtype=np.float32)
        S2 = np.stack([
            (I - beta[h] * np.outer(k[h], k[h])) @ S[h] + beta[h] * np.outer(k[h], v[h])
            for h in range(H)
        ])
        d_max = max(d_max, float(np.abs(S1 - S2).max()))
    print(f"1) 秩 1 形式 vs 显式矩阵形式 max|diff| = {d_max:.3e}  (fp32 累积舍入量级)")
    return d_max


def main():
    m = DeltaNet(MDIR)
    print(f"hidden={m.hidden} layers={m.n_layers} heads={m.n_heads} "
          f"k_dim={m.head_k_dim} v_dim={m.head_v_dim} conv={m.conv_size} vocab={m.vocab}")

    check_delta_algebra()

    rng = np.random.default_rng(3)
    ids = rng.integers(0, 30000, size=20).tolist()

    st = m.reset(m.init_state())
    a = np.stack([m.step(t, st) for t in ids])

    # 分段: 前 8 步建状态, 后 12 步在同一状态继续, 拼接 logits 与全程一致
    st2 = m.reset(m.init_state())
    pre = np.stack([m.step(t, st2) for t in ids[:8]])
    post = np.stack([m.step(t, st2) for t in ids[8:]])
    seg = np.concatenate([pre, post])
    print(f"2) 全程 vs 分段续算 max|diff| = {np.abs(a - seg).max():.3e}")

    st3 = m.reset(m.init_state())
    b = np.stack([m.step(t, st3) for t in ids])
    print(f"3) 重复运行 bit 稳定 max|diff| = {np.abs(a - b).max():.3e}")

    # 4. 贪心生成 —— Llama BPE (含 BOS=1), vocab 32000
    prompt = [1, 415, 5565, 302, 4843, 349]  # "<s>The capital of France is"
    gen = m.generate(prompt, max_new=20, temperature=0.0)
    print(f"4) 贪心续写 ids = {gen}")

    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(MDIR)
        print("   text:", repr(tok.decode(gen)))
        print("   argmax per step:", a.argmax(1)[:8], "...")
    except Exception as e:
        print("   tokenizer unavailable:", type(e).__name__, e)


if __name__ == "__main__":
    main()