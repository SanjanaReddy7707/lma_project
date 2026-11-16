# Make-up Mid-Sem — CL3410 (Variant 3)

Roll number ends in 3 → **variant 3** (3 mod 5 = 3).
Part A: **Model H (Hindi)**, LoRA on **W_Q, W_V in every block**, DPO **β = 0.05** (and β = 0.1).

All commands are run from `lm_project/` with the project venv:

```bash
source venv312/bin/activate        # torch 2.x, sentencepiece, matplotlib, transformers (Part C only)
```

## Part A — LoRA + DPO on the project Model H checkpoint

Starts from `phase2/checkpoints_H/best.pt` (pretrained Model H) and uses the Phase 3
Hindi reasoning data in `phase3/data/hindi_reasoning/`. No `peft`, no DPO trainer.

```bash
python midsem/partA/build_data.py   # A0/A3: seen-entity + unseen-wording test sets, 2,000 preference pairs
python midsem/partA/train_lora.py   # A1/A2: step-0 logit check, param counts, r = 1/4/16 sweep, memory
python midsem/partA/train_dpo.py    # A4/A5: DPO β = 0.1 and β = 0.05, margin plot, drift (val PPL)
```

| File | Content |
|---|---|
| `midsem/partA/lora.py` | `LoRALinear` (frozen W0, trainable A/B, α/r scaling) and `apply_lora_qv` |
| `midsem/partA/train_dpo.py` | `seq_logprob`, `dpo_loss`: DPO loss written by hand |
| `midsem/common.py` | model loading, EM evaluation (same rule as Phase 3), pretraining-val PPL |
| `midsem/data/preference_pairs.jsonl` | 2,000 preference pairs (training split only) |
| `midsem/data/test_seen_entities.jsonl`, `test_unseen_wording.jsonl` | new test sets; the unseen-entities set is the original `phase3/.../test.jsonl` |
| `midsem/results/partA_lora.json`, `partA_dpo.json` | every number in the report |
| `midsem/figures/dpo_margin.png` | margin + loss curves for both β |

## Part C — reasoning and judging (Qwen2.5-1.5B-Instruct, judge Phi-3.5-mini-instruct)

```bash
python midsem/partC/checker.py      # self-test: 0.05 == 1/20 == 5E-02
python midsem/partC/problems.py     # writes midsem/data/c1_problems.jsonl (20 Hindi problems, 5 modified)
python midsem/partC/run_c1.py       # direct / zero-shot CoT / self-consistency@5
python midsem/partC/run_c2.py       # 30 pairs × 2 orders, judge metrics
```

Outputs: `midsem/results/c1_generations.jsonl`, `c1_summary.json`,
`midsem/data/c2_pairs_with_verdicts.jsonl`, `midsem/results/c2_summary.json`.

## Figures and report

```bash
python midsem/make_figures.py       # partA_accuracy.png, c1_acc_vs_tokens.png (dpo_margin.png comes from train_dpo.py)
python midsem/make_report.py        # → midsem/report.pdf (report.md + partB.md + ai_note.md + figures)
```

## What else the code needs

The scripts import my project code unchanged, so they run from inside `lm_project/`:
`phase2/model.py`, `phase3/config_ft_H.py`, `phase3/reasoning_dataset.py`,
`phase3/generate_hindi_reasoning.py`, `phase3/data/hindi_reasoning/`, `hindi/tokenizer/hindi_bpe.model`,
and `data/hindi/splits/val_hindi_bpe_ctx256.bin` (for perplexity).
The two checkpoints are too large for Moodle (297 MB each): `phase2/checkpoints_H/best.pt` (pretrained, Drive link in
the project README) and `phase3/checkpoints_ft_H/best_ft.pt` (Phase 3 full finetune, used only for the comparison row).
Part C downloads `Qwen/Qwen2.5-1.5B-Instruct` and `microsoft/Phi-3.5-mini-instruct` from Hugging Face on first run.

Hardware used: Apple M4 Pro (MPS backend), 48 GB unified memory.
