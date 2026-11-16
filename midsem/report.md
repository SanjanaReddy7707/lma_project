**Variant 3** (roll number ends in 3; 3 mod 5 = 3)

# CL3410 Make-up Mid-Semester — Report

Part A: Model H (Hindi, 24.4 M params, 12 layers, d = 384), LoRA on W_Q and W_V in every block, DPO β = 0.05 and 0.1. Part B: b = 40, r = 3; C = 2×10^21; 4096 × 11008, r = 8; 7 B; α = 0.2. Part C: Qwen2.5-1.5B-Instruct, judge Phi-3.5-mini-instruct. Every number below is in `midsem/results/*.json`; commands are in `midsem/README.md`.

## Part A — LoRA and DPO on Model H

**Test sets.** Phase 3 had one held-out test set (6 names never seen in training), which I use as *unseen entities*. I built two more with `build_data.py`: *seen entities* (200 new questions from the same 8 templates with the 20 training names, any input identical to a training input removed) and *unseen wording* (200 questions with training names but context sentences and question phrasings that never occur in `train.jsonl`, e.g. "की तुलना में", "इनमें से", "के मामले में सबसे ऊपर कौन है?"; the script asserts each new fragment is absent from the training text). Scoring is the Phase 3 exact-match rule. The pretrained model scores 0 on all three.

**A1 — LoRA from scratch** (`partA/lora.py`). `LoRALinear` keeps W0 frozen and adds (α/r)·B A x. A ∈ ℝ^{r×384} uses Kaiming-uniform init, B ∈ ℝ^{384×r} is zero, α = 2r. My model fuses Q, K, V into one `qkv_proj` (384 → 1152), so the wrapper takes row slices: rows 0–383 (Q) and 768–1151 (V) each get their own (A, B), and rows 384–767 (K) get none. This is exactly equivalent to separate adapters on W_Q and W_V. Everything else is frozen.

Trainable parameters = 12 blocks × 2 matrices × r × (384 + 384) = 18,432 r: **18,432 (0.075 %)** at r = 1, **73,728 (0.30 %)** at r = 4, **294,912 (1.19 %)** at r = 16, out of 24.45 M (+ adapters).

**Step-0 check:** on 8 validation questions the wrapped model's logits equal the pretrained model's exactly: `torch.equal` is True and the max |Δ| is 0.0. Why: B = 0, so B(Ax) is a tensor of exact zeros (0 × finite = 0 in floating point), and W0x + 0 = W0x bit-for-bit. Both models are in eval mode, so dropout is off. A is non-zero (max |A| = 0.051), so the gradient with respect to B is non-zero and training can start.

**A2 — Rank sweep.** Each rank starts from `phase2/checkpoints_H/best.pt` and trains on the Phase 3 training set: AdamW lr 1e-3, batch 32, 1,000 steps, cosine schedule, answer-only loss. I keep the step with the best validation EM (validation uses 6 more held-out names), which is the same selection rule as Phase 3.

| Model | trainable | val EM | seen ent. | unseen ent. | unseen wording |
|---|---|---|---|---|---|
| Phase 3 full FT | 24,446,208 | 0.40 | 0.540 | 0.355 | 0.235 |
| LoRA r = 1 | 18,432 | 0.570 | **0.560** | 0.490 | **0.280** |
| LoRA r = 4 | 73,728 | **0.600** | 0.505 | **0.575** | 0.240 |
| LoRA r = 16 | 294,912 | 0.565 | 0.480 | 0.490 | 0.195 |

| Memory (same batch, after first step) | Full FT | LoRA r = 4 |
|---|---|---|
| AdamW state (m, v) | 195.6 MB (24.4 M × 2 × 4 B) | **0.59 MB** (73,728 × 2 × 4 B) |
| Allocated device memory (MPS) | 478.5 MB | 137.0 MB |

What these show:
- **LoRA matches or beats full FT.** It is better on unseen entities at every rank, and r = 1 beats it on all three sets.
- **Rank does not help.** r = 16 is the worst. The data are 5,000 sentences from 8 templates, so a rank-1 update of Q and V is enough to learn the answer format and the comparison. Extra rank mostly fits the training names: r = 16 reaches its best validation EM at step 200 and then degrades.
- **Unseen wording is the real gap** (0.20–0.28 for every model). Per template, r = 4 gets T2 19/19 and T7 21/21 on unseen entities but **0/19 and 0/15** when the same logic is phrased with new words, and T6 sorting is 0 everywhere. The models learnt the surface templates, not the comparison.
- **r = 4 scores higher on unseen than on seen entities.** Per template, the two sets are within a few points of each other (T1 17/41 vs 18/42, T7 14/14 vs 21/21). The gap comes from the different template mix plus noise: with n = 200 the standard error is about ±3.5 points.
- **Caveat on the comparison.** LoRA used a 50× larger learning rate and twice the steps of Phase 3, so this is not an equal-budget comparison.
- **Memory.** The optimiser state is 330× smaller. Total memory falls only about 3.5× because the frozen weights (98 MB), activations and the MPS cache still dominate. MPS has no peak-memory counter, so I report memory allocated after step 1.

**A3 — Preference pairs** (`data/preference_pairs.jsonl`, 2,000 pairs). Each pair is a training question, the gold answer as *chosen*, and a plausible wrong answer as *rejected*:
- **other entity** (1,620 pairs): the other or a wrong name in the same sentence, e.g. "अर्जुन के पास कम सेब हैं।" → "रमेश के पास …". These only differ in the name.
- **swapped order** (206 pairs, sorting template): two neighbours in the ordering swapped.
- **false inequality** (174 pairs, equal-values template): "X की गति अधिक है" instead of "दोनों बराबर".

How test data was kept out:
- **(i) Source.** All pairs are sampled from `train.jsonl` only, and rejected names come from the same question.
- **(ii) Names.** The Phase 3 generator draws train, validation and test names from disjoint lists, and I assert that no validation or test name appears in any prompt, chosen or rejected reply.
- **(iii) Inputs.** I assert that no prompt equals any input of the three test sets.
- **(iv) Wording.** I assert that none of the new unseen-wording fragments appears in any prompt.

All pairs are in Hindi with the project tokenizer.

**A4 — DPO from scratch** (`partA/train_dpo.py`). log π(y|x) is the sum of token log-probabilities over the reply tokens only (reply + "\n" + EOS; the prompt is masked), in the same format as SFT. The loss is

L = −log σ( β [ (log π_θ(y_w|x) − log π_ref(y_w|x)) − (log π_θ(y_l|x) − log π_ref(y_l|x)) ] ).

The policy and the frozen reference are both the r = 4 LoRA model; only A and B are trained. Settings: 300 steps of 16 pairs, lr 1e-4.

**Step-0 loss = 0.693147 for both β, which is exactly log 2.** At step 0 the policy and the reference are the same network, and I keep the policy in eval mode so dropout cannot make them differ. Both log-ratios are therefore 0, the margin is 0, and L = −log σ(0) = −log ½ = log 2, independent of β.

| after 300 steps | β = 0.1 | β = 0.05 |
|---|---|---|
| margin (chosen − rejected log-ratio) | 5.44 | 6.78 |
| chosen log-ratio / rejected log-ratio | −0.86 / −6.30 | −1.26 / −8.03 |
| seen / unseen ent. / unseen wording | 0.450 / 0.490 / 0.195 | 0.445 / 0.480 / 0.185 |
| (start: LoRA r = 4) | 0.505 / 0.575 / 0.240 | |

What the runs show:
- **Margins grow quickly in both runs** (Fig. 1). The smaller β ends with the larger margin: the loss depends only on β·margin, so β = 0.05 has to open a gap about twice as wide to reach a similar loss. It also sits further from the reference.
- **The curves move together** because both runs draw the identical batch sequence. The jaggedness is batch-to-batch noise from 16-pair batches.
- **DPO lowered test accuracy on every set, more so for the smaller β.** The margin grows mostly by pushing the rejected reply down, but the chosen log-ratio also falls (−0.86, −1.26). Greedy exact match needs the gold reply to be the most likely one, and that is exactly what DPO does not protect: it optimises the difference, not the likelihood of the gold reply.
- **The pairs add little.** Chosen and rejected share every token except the name, so the gradient concentrates on one token. That token is already what SFT trained on with the same labels, so DPO adds no new information here.

**A5 — Drift** (perplexity on the Hindi pretraining validation split, same protocol as Phase 2; my base number reproduces the Phase 2 report's 38.86 exactly):

| base | Phase 3 full FT | LoRA r = 4 | DPO β = 0.1 | DPO β = 0.05 |
|---|---|---|---|---|
| 38.86 | 44.47 | 50.22 | 51.07 | 51.50 |

Every adaptation moves the model away from general Hindi text, because it trains only on templated "प्रश्न/उत्तर" sentences.
- **LoRA drifted more than full FT.** This is the opposite of the usual "LoRA forgets less" finding, and I attribute it to my training budget, not to the method. LoRA ran at lr 1e-3 for 1,000 steps, against 2e-5 for 500. A low-rank change to W_Q and W_V in all 12 blocks still re-routes attention on every token.
- **DPO adds a little more drift, and more for the smaller β.** DPO's implicit KL constraint keeps the policy near the *reference*, which is the already-drifted LoRA model, not the pretrained base. A smaller β weakens that constraint, so the policy moves further.

{{PARTB}}

## Part C — Reasoning and judging

**C1 setup.** I wrote 20 Hindi problems (`data/c1_problems.jsonl`): 15 standard and 5 modified. Each modified problem copies a standard one and changes one word, e.g. "दे दिए" → "से ले लिए" (gave → took), or the train "खड़ी रही" (stood still) for 3 hours.

Conditions on Qwen2.5-1.5B-Instruct (fp16):
- **direct**: greedy, "only the number".
- **zero-shot CoT**: greedy, "चरण-दर-चरण सोचिए … 'उत्तर: <संख्या>'".
- **SC@5**: the CoT prompt with T = 0.7, top-p 0.95, majority vote over normalised answers; ties go to the earliest sample.

**Checker** (`partC/checker.py`). Answers are parsed to exact `Fraction`s, so 0.05, 1/20 and 5E-02 are equal, and Devanagari digits and thousands commas are handled. The answer is the number inside `\boxed{}` if present, else the first number after the last "उत्तर"/"answer", else the last number. The self-test asserts these cases. I hand-checked the raw outputs of every surprising result (S01, S03, S04, S07, S14): each was a genuine model error, not a parsing error.

| | standard (15) | modified (5) | all | mean generated tokens | gave template answer on modified |
|---|---|---|---|---|---|
| direct | 0.33 | 0.20 | 0.30 | 4.9 | 1/5 |
| zero-shot CoT | 0.67 | 0.40 | 0.60 | 234 | 2/5 |
| SC@5 | **0.80** | 0.40 | **0.70** | 1,474 (295/sample) | 2/5 |

**Thinking longer helps on standard problems** (0.33 → 0.67 → 0.80, Fig. 3). The cost is about 50× the tokens for CoT and 300× for SC. **It does not help on the modified problems.** CoT and SC both stop at 2/5, and the share giving the familiar template answer *rises* from 1/5 to 2/5:
- **M02** (took apples): 7 instead of 17.
- **M03** (train standing still): 180 instead of 0, in all three conditions.

Longer reasoning made the model run the memorised template more confidently; it did not read the changed word. The drifting-boat problem M01 failed everywhere. CoT used (10 − 2) × 3 = 24, the upstream formula: it applied a boat formula without noticing the engine is off.

**Finding on Hindi ability:** all 20 CoT outputs stayed in Hindi, but many are incoherent:
- **S03:** degenerate repetition of "यह भावना है कि छात्रों को बाँटा गया है".
- **S04:** never converts "3 दर्जन" to 36.
- **S14:** five SC samples give five different wrong answers.

At this scale the model's Hindi arithmetic is the bottleneck, and SC helps mainly by out-voting such noise.

**C2 setup.** I made 30 pairs from the C1 outputs (`data/c2_pairs_with_verdicts.jsonl`). Each pairs the direct answer with one CoT answer (the greedy CoT or an SC sample) for the same problem, at most 2 per problem. By checker label:

| short right, long wrong | short wrong, long right | both right | both wrong |
|---|---|---|---|
| 6 | 10 | 6 | 8 |

Only 6 distinct short-right/long-wrong pairs existed under the 2-per-problem cap. The judge is Phi-3.5-mini-instruct (Microsoft, a different family from Qwen), greedy, asked for "A or B". Every pair was run in both orders.

| metric | value |
|---|---|
| position-bias rate (verdict flips with order) | **8 / 30 = 0.27** |
| order-consistent pairs | 22 (12 with a unique checker winner) |
| agreement with checker (those 12) | **8 / 12 = 0.67** |
| Cohen's κ (those 12) | **0.00** |
| longer answer wins (all 60 verdicts) | **40 / 60 = 0.67** |
| short-right / long-wrong pairs: judge picks short in both orders | **0 / 6** (picks the wrong long one in both orders: 4 / 6) |

**κ = 0 despite 67 % agreement.** On all 12 order-consistent pairs with a unique winner, the judge chose the CoT answer. The checker labels 8 of them CoT and 4 direct, so the observed agreement is 8/12, and the agreement expected by chance from those label rates (1 × 8/12) is also 8/12. κ = (p_o − p_e)/(1 − p_e) = 0. The judge's decisions carry no information beyond "pick the long one". The short-right/long-wrong row shows it directly: a correct one-number answer never beat a wrong derivation.

**Verdict.** I would not use this judge to score C1: it flips on 27 % of pairs when the order is swapped, prefers length (κ = 0 against the checker), and lost every case where the short answer was right. For single-number answers the checker should be used instead. If a judge is needed, the first change is to give it the reference answer, ask for the extracted final number before any verdict, and average both orders.

## Note on AI tool use

{{AINOTE}}

## Appendix — Figures

Fig. 1 — DPO: mean chosen − rejected log-ratio margin and loss, β = 0.1 vs 0.05 (both start at log 2).
{{FIG:dpo_margin.png}}
Fig. 2 — Exact match on the three test sets for every Part A model.
{{FIG:partA_accuracy.png}}
Fig. 3 — C1 accuracy against mean generated tokens (log scale).
{{FIG:c1_acc_vs_tokens.png}}
