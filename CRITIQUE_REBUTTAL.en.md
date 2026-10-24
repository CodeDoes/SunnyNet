*[中文版](CRITIQUE_REBUTTAL.md)*

# Line-by-line check of external critiques

Against a set of external criticisms (short-sequence inversion / scaling law failure / weak long-range retrieval / random datasets /
poor multi-seed generalization / T=1024 slowest and highest memory), checked line by line against the repository's existing measured data and newly added measurements,
classified into "Supported", "Not supported", "Insufficient evidence".

New scripts and data: `experiments/_T1024_audit.py`, `prepare_extra.py`, `experiments/_ood.py`.

---

## 1. Line-by-line judgments

| Critique | Judgment | Basis |
|---|---|---|
| T<384 slower than Transformer | Supported, but only for the batch=1 forward criterion | §2.4 crossover T≈384; training mode (batch=8) crossover moves down to T<256 (§19) |
| Therefore limited to conventional NLP tasks | Not supported (premise wrong) | Short-sentence classification / dialogue replies are training/fine-tuning scenarios; training mode T=256/384 already 1.5–2.4× faster (§19) |
| Scaling law fails | Insufficient evidence | This repository has only tested two points, 20M / 100M (§14.6 / §14.8), no scaling curve, cannot judge "failure" |
| Weak long-range retrieval | Not supported for the code domain, supported for the natural language domain | §8: code domain far-bucket hit 0.1208, consistent with Transformer; natural language domain near 0 |
| Measured on random datasets | Supported only for speed benchmarks, not for quality conclusions | Speed scripts use `torch.randint` (compute graph independent of values); quality comparisons go through `DomainMixer`→real token stream (§3/§10/§12) |
| Large multi-seed variance | Supported, and known to the project | §6: seed ppl range 6.32 > architecture difference 3.53; §10.5 already states a single seed cannot decide |
| Poor cross-dataset generalization | Not supported | Chinese new domain rank 1/7, English new domain rank 2/7, OOD drop lowest tier (§18) |
| T=1024 slowest | Premise-dependent: batch=1 forward ranks 5th (not slowest, TF last); training mode ranks 1st | §10.6 / §17 |
| T=1024 highest peak memory | Not supported | §17: SunnyNet 1812 MB ranks 2nd lowest tier, saves 43% over Transformer |

---

## 2. T=1024 training-mode speed and memory (§17)

The existing `on_scaling_t8.json` only has batch=1 pure forward, and never recorded memory. Supplementary training-mode measurement:

Speed (ms/token): sunnynet **0.13186 (1st)** · h3 0.13190 · mamba 0.14721 ·
hyena 0.16754 · linear_attn 0.17646 · rwkv 0.19677 · transformer 0.55693

Peak memory (MB): transformer 3192 · **sunnynet 1812 (2nd)** · hyena 1809 ·
mamba 1780 · linear_attn 1777 · h3 1689 · rwkv 1611

⇒ In training mode neither allegation holds. Forward and training-mode rankings differ (backward accounts for 56.9% of the training step, §13.3).

---

## 3. Training-mode check of short-sequence inversion (§19)

Retested per training mode (batch=8, including backward) with three configs × three lengths × 3 rounds:

| Config | T=128 | T=256 | T=384 |
|---|---|---|---|
| `small` (old d128) ratio to TF | 1.104 | 0.672 | 0.581 |
| `smalln80` (narrow, §16) ratio to TF | 1.045 | 0.509 | 0.425 |

(ratio <1 = faster than Transformer)

- Crossover moves down from T≈384 (forward) to T<256 (training mode): at T=256/384 already 1.5–2.4× faster.
- T=128 still slower: old config 10.4%, narrow config 4.5% (within the high-memory-pressure noise band).
- The effective lever is reducing activation width (narrow residual stream): T=128 disadvantage narrows from 10.4% to 4.5%,
  T=256/384 speedup 24–27%. Widening or tuning batch does not work (§16.5/§13.3: the root cause is operator count).

---

## 4. Cross-distribution (OOD) generalization

Method: original three domains (zh_ancient/en_literature/code_python) trained 600 steps,
ppl measured on external new datasets not seen during training.

New datasets (`prepare_extra.py`, independently collected, not splits of the original corpus):
- `arxiv_en`: 500 English scientific abstracts from the arxiv API, 236,595 tokens
- `zh_tech_news`: 97 technology/science article bodies from Xinhua/Gmw/People.cn, 164,711 tokens

Fairness premise: the BPE vocab comes from traditional-Chinese ancient texts, `zh_tech_news` unregistered-character rate 23.02% (many simplified characters fall outside the vocab),
`arxiv_en` only 0.05%. The high ppl of the Chinese new domain contains a tokenizer-mismatch component, not all distribution shift,
hence the two are stated separately, not merged into one ratio.

### Results (lower ppl is better; 600 steps, single seed, T=1024)

| Architecture | arxiv_en (new) | zh_tech_news (new) | ext/orig drop |
|---|---|---|---|
| **sunnynet** | **35.58 (2nd)** | **321.08 (1st)** | **1.480** |
| h3 | 31.52 (1st) | 446.77 (4th) | 1.840 |
| rwkv | 41.03 (4th) | 427.22 (3rd) | 1.691 |
| mamba | 40.87 (3rd) | 499.74 (6th) | 1.964 |
| hyena | 54.22 (5th) | 400.39 (2nd) | 1.481 |
| linear_attn | 61.55 (7th) | 489.23 (5th) | 1.694 |
| transformer | 61.34 (6th) | 531.77 (7th) | 1.817 |

### Reading

- "Poor cross-dataset generalization" not supported: Chinese new domain SunnyNet 1/7 (that domain puts it at a disadvantage with 23% unk);
  English new domain 2/7; OOD drop 1.480, lowest tier among the seven architectures (tied with hyena).
- Criterion trap: `zh_ancient`'s "in-domain" ppl is the highest across all architectures (sunnynet 326, transformer 424),
  because 600 steps did not converge + that domain (traditional-Chinese ancient texts) is the hardest. Using it as the denominator to compute a ratio is distorted—
  `sunnynet`'s `news/zh_anc=0.985` looks like "no drop", actually the denominator is inflated,
  cannot be read as "cross-domain better than in-domain". The only reliable metric is `arxiv_en/en_literature`:
  sunny 1.706, same tier as h3 1.686 / hyena 1.691.
- Cannot counter-claim "strong generalization": absolute ppl is still high, 600 steps not converged, single seed.
  The only reliable conclusion: cross-distribution is not worse than peers of the same complexity.

(Data in `experiments/results/_ood.json`.)

---

## 5. Two places in the critiques where "the conclusion is right but the attribution is wrong"

1. "Short-sequence inversion limits conventional NLP applications": the direction is right, but the premise is forward latency.
   Training mode T=256 already catches up with rwkv (§16), and §16.5 already proves the parity holds only at L=4—
   when layers are aligned to L=6, the same-budget narrow config is instead 30–40% slower, the root cause is operator count not FLOPs.
   The correct direction of mitigation is reducing operator count (fusion/fewer layers), not widening or tuning batch.

2. "Scaling law fails": this statement has no data support in the report—
   this repository only measured two points, 20M / 100M, the TL;DR explicitly writes "the scaling trend is not monotonic",
   which itself shows a scaling curve has not yet been established, let alone "failure". Judging requires ≥4 scale points.

---

## 6. Conclusion

- Supported: T<384 forward inversion (measured, structural); large seed variance (known);
  natural language domain long-range retrieval near 0 (measured).
- Not supported: poor cross-dataset generalization (OOD measured Chinese new domain 1st, English new domain 2nd, lowest-tier drop);
  T=1024 training mode "slowest + highest memory"; "measuring quality with random data" (quality uses real corpus);
  code domain "weak long-range retrieval" (on par with Transformer).
- Insufficient evidence: scaling law failure (this repository has no scaling curve, only the 20M/100M two points).
- New: T=1024 training-mode audit (§17); OOD cross-domain measurement (§18, overturns "poor generalization").

Statements unsupported by data: any "comprehensively faster", "stronger as scale grows", "long-range stronger than Transformer",
"strong generalization capability" (only "cross-distribution is not worse than peers" can be said).