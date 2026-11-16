## Part B — Calculations and design (variant 3)

### B1. Near-duplicate detection (b = 40, r = 3, 120 hashes)

A pair with Jaccard s agrees on one MinHash with probability s; on a whole band of r rows with s^r; it becomes a candidate if at least one of b bands agrees:
P(s) = 1 − (1 − s^r)^b.

- s = 0.8: s^3 = 0.512; 1 − 0.512 = 0.488; 0.488^40 = 3.4 × 10^-13 → **P ≈ 1.0000** (misses ≈ 3 in 10^13).
- s = 0.3: s^3 = 0.027; 0.973^40 = e^(40·ln 0.973) = e^(−1.095) = 0.335 → **P ≈ 0.665**.
- Threshold (steepest point of the S-curve) t ≈ (1/b)^(1/r) = (1/40)^(1/3) = 0.025^(1/3) ≈ **0.29**.

Would I use it? **No, not as is.** A threshold of 0.29 is far below the 0.7–0.8 usually meant by "near duplicate". Two-thirds of pairs that only share 30 % of their shingles become candidates. In Hindi or Nepali news that covers every pair of articles about the same event, and every pair that shares boilerplate (headers, "यह भी पढ़ें…", copyright lines). The costs:
- If I *verify* candidates with exact Jaccard, recall is near-perfect but the candidate set blows up. Verification costs O(#candidates), and with a 0.29 threshold that approaches the all-pairs cost LSH is meant to avoid.
- If I *drop* candidates without verifying, I throw away many genuinely different documents. For a low-resource corpus that is the expensive mistake: tokens are the scarce resource. It also removes topical diversity (all reporting on one event collapses to one article).

With the same 120 hashes I would use b = 15, r = 8 (t = (1/15)^(1/8) ≈ 0.71): P(0.8) = 0.94, P(0.5) = 0.06, P(0.3) = 0.001. The price is losing about 6 % of the 0.8-similar pairs, so a few duplicates survive, against a candidate set two to three orders of magnitude smaller. For a small corpus I prefer leaving some duplicates in to deleting real text.

### B2. Mixing two languages (α = 0.2)

Phase 1 training-split token counts, measured by the project tokenizers (length of the `train_*_bpe_ctx256.bin` int32 caches):
H (Hindi) N_H = 227.2 M tokens, L (Nepali) N_L = 372.2 M tokens. (The Phase 1 report's "~666 M / ~1,397 M" were pre-tokenizer *estimates*. Model L's language has *more* tokens than Model H's. "Low-resource" in this project refers to the language, not to my corpus size.)

Sampling with temperature α: p_i = N_i^α / Σ_j N_j^α.
- By size (α = 1): p_H = 227.2 / 599.4 = **0.379**, p_L = **0.621**.
- α = 0.2: 227.2M^0.2 = e^(0.2·19.241) = 46.86; 372.2M^0.2 = e^(0.2·19.735) = 51.72; sum = 98.58 → p_H = **0.475**, p_L = **0.525**.

α < 1 flattens the distribution towards uniform: H rises from 38 % to 48 %, and L (the bigger corpus here) falls from 62 % to 52 %.

Budget 500 M tokens: L receives 0.525 × 500 M = 262 M tokens → 262 / 372.2 = **0.70 epochs** over the Model L corpus (H: 237.7 M / 227.2 M = 1.05 epochs). With the Phase 1 report's estimated counts the answer would be 0.19 epochs over L; either way it is below one.

Repetition: Muennighoff et al. ("Scaling data-constrained LMs") find up to ~4 epochs of repeated data is almost as good as fresh data, with value decaying after that and near zero by ~16 epochs. At 0.7 epochs nothing is repeated at all, so the mixture does not hurt Nepali through over-repetition. If anything it *under*-uses the Nepali data. (For reference, my separate Model L actually saw 12,000 × 256 × 256 = 786 M tokens ≈ 2.1 epochs.)

Was keeping them separate the better call? **For this project, yes.** (1) Each model was ~25 M parameters. A shared model would split that capacity across two languages and need a joint vocabulary. With 8 k BPE per language, merging two 8 k vocabularies roughly doubles the embedding table (≈3 M parameters each at d = 384), eating ~12 % of the budget. (2) The project needed per-language comparisons of tokenizer fertility, perplexity and finetuning, and separate models keep those clean. (3) The usual argument for mixing is transfer to a data-starved language. Here the "L" language has more data than H, and Hindi and Nepali already share the script, so the transfer gain would be small. A bilingual model would be worth it at larger scale, or if Nepali really had ~10× less data. Then α = 0.2 would up-sample it, and repeating ≤ 4 epochs would be cheap.

### B3. Compute-optimal allocation (C = 2 × 10^21 FLOPs)

C ≈ 6ND and D = 20N ⇒ C = 120 N² ⇒ N = √(C/120) = √(1.667 × 10^19) = **4.08 × 10^9 parameters (≈4.1 B)**, D = 20N = **8.16 × 10^10 tokens (≈82 B)**. Check: 6 × 4.08e9 × 8.16e10 = 2.0e21 ✓.

Colleague: N' = 4N = 16.3 B, same C ⇒ D' = C / (6N') = D/4 = 20.4 B tokens, i.e. only **1.25 tokens per parameter**.

In L(N, D) = L∞ + A/N^α + B/D^β:
- **A/N^α improves**: it falls by a factor 4^α (Chinchilla fit α ≈ 0.34: ×0.62).
- **B/D^β gets worse**: it rises by 4^β (β ≈ 0.28: ×1.47).
- L∞ is unchanged.

Plugging in the Hoffmann et al. constants (E = 1.69, A = 406.4, B = 410.7, α = 0.34, β = 0.28): compute-optimal 0.219 + 0.362 → L = 2.271; colleague 0.137 + 0.533 → L = 2.360. The colleague is worse by ~0.09 nats.

Why: under a fixed C, N and D trade off one-for-one in FLOPs. The 20 tokens/param point is where the marginal gain from more parameters equals the marginal gain from more data. Moving 4× away from it buys a small drop in the parameter term and pays a larger rise in the data term, because the data term is the bigger of the two at this point. At 1.25 tokens/param the model is badly under-trained. The only way the colleague is right is if *inference* cost matters more than training cost; then you would go the other way (smaller N, more D), not bigger.

### B4. LoRA and memory (W ∈ ℝ^{4096×11008}, r = 8; 7 B model)

ΔW = BA with B ∈ ℝ^{4096×8}, A ∈ ℝ^{8×11008}.
Trainable = r(d + k) = 8 × (4096 + 11008) = 8 × 15,104 = **120,832**.
Full update = d·k = 4096 × 11008 = **45,088,768**. Share = 120,832 / 45,088,768 = **0.268 %** (≈1/373).

Initialisation: B = 0 and A random (Kaiming/Gaussian). Then BA = 0, so at step 0 the adapted model *is* the pretrained model (my Part A1 check confirms bit-identical logits). Training starts from the known-good function, not from a randomly perturbed one. A must be non-zero so there is a gradient: ∂L/∂B = (∂L/∂h)(Ax)ᵀ is non-zero because Ax ≠ 0. After the first step B ≠ 0 and then A gets gradient too.
**If both start at zero**: ∂L/∂B ∝ Ax = 0 and ∂L/∂A ∝ Bᵀ(∂L/∂h) = 0. Both gradients are exactly zero, and they stay zero forever, so the adapter never learns (a saddle point at the origin). Plain SGD/Adam has no noise to escape it, and weight decay keeps it at zero.

Memory, 7 B parameters, fp32, before activations:
- **Inference**: 7 × 10^9 × 4 B = **28 GB** (26.1 GiB).
- **Full finetune with Adam**: weights 4 B + gradients 4 B + Adam m 4 B + Adam v 4 B = 16 B/param → 7 × 10^9 × 16 = **112 GB** (104 GiB). (Mixed-precision recipes with an fp32 master copy are similar, ~16–18 B/param.)
- For comparison, LoRA r = 8 on all 7 B weights' attention matrices adds only tens of MB of gradient + optimiser state on top of the 28 GB frozen weights.

### B5. Loss masking (3 user + 3 assistant turns)

Standard practice: **loss on all three assistant turns, including their end-of-turn tokens; no loss on system/user tokens or on the role-header template tokens.** User text is input we condition on, not behaviour we want to imitate. Training on it would teach the model to write like users (and to keep going after its own turn, hallucinating the next user message). The middle assistant turn, "could you share your order number?", is *correct* assistant behaviour and must be scored like any other. The end-of-turn token in each assistant turn is what teaches the model to stop and hand control back.

If you score only the final assistant turn, the model never gets a gradient for the turn where the right move is to **ask a clarifying question** (request the missing order number) instead of answering. The supervised signal is then all "final answer" turns, so the model learns to always answer immediately and hallucinate order details instead of asking. You also throw away 2/3 of the assistant tokens per conversation.

### B6. Picking an alignment method

(i) **Warmer Hindi support replies, 5,000 human preference pairs → DPO.** The data is pairwise preferences over a subjective, hard-to-specify quality (tone). That is exactly what DPO consumes directly: no separate reward model, no RL loop, stable and cheap on 5 k pairs. PPO would first need a reward model trained on the same 5 k pairs, which would likely be noisy and easy to hack (e.g. by inserting "धन्यवाद" everywhere). *Reward role*: the implicit reward β·log(π_θ/π_ref) of the preferred reply, fitted through the Bradley–Terry likelihood of the human choices.

(ii) **Python functions that must pass a test file → GRPO.** The reward is verifiable and automatic: run the tests. So we can sample several completions per prompt and learn from them on-policy. GRPO uses the group mean as the baseline, so we skip PPO's value network, which is hard to train for sparse pass/fail rewards. DPO would need fixed preference pairs and cannot explore new solutions. *Reward role*: the unit-test result (fraction of tests passed, or 1/0 pass-all), possibly plus a compile/format penalty.

(iii) **Marathi grade-school word problems with a single numeric answer → GRPO.** Again a verifiable reward (exact numeric match, checked by a checker like my C1 one). Sampling a group per problem gives relative advantages without a critic, as in DeepSeekMath. It also encourages correct reasoning chains rather than imitating fixed solutions, which helps when Marathi supervised data is scarce. *Reward role*: 1 if the extracted final number equals the gold answer (0 otherwise), optionally plus a small format reward for writing the answer in the requested place.
