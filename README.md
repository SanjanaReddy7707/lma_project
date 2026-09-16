# LMA Project — Building Monolingual Transformer LMs in Two Indian Languages

**Languages:**  
- **Hindi (hi)** — higher-resource model (Model H)  
- **Nepali (ne)** — lower-resource model (Model L)  

**Task:** Build two decoder-only Transformer language models (~25M parameters each) from scratch.

---

## Phase 2: Model Implementation, Pretraining & Evaluation

### Model Results

| | Model H (Hindi) | Model L (Nepali) |
|---|---|---|
| Parameters | 24,446,208 | 23,130,112 |
| Architecture | 12 layers, d=384, 6 heads | 6 layers, d=512, 8 heads |
| Training steps | 6,500 | 12,000 |
| Val loss | 3.6600 | 3.8211 |
| Val PPL | 38.86 | 45.65 |
| Val BPB | 0.5362 | 0.4635 |
| Test PPL | 38.96 | 46.38 |
| Test BPB | 0.5400 | 0.4661 |

### Google Drive Links — Phase 2

> **Model H checkpoint (best.pt)**:https://drive.google.com/file/d/14q--XzhKfME7aUWoAUR80nVWXyMCcm2I/view?usp=drive_link 
> **Model L checkpoint (best.pt):** (https://drive.google.com/file/d/19IfW2_GGfpy1uoSGqRKKphRbtZinrOhb/view?usp=drive_link) 
> **Phase 2 Report PDF:** https://drive.google.com/file/d/1ZeqdMhDxaGkeJFieTgNk3B2Y4lU_c1k8/view?usp=sharing

### Phase 2 Reproduction Steps

```bash
# 1. Install dependencies
source venv312/bin/activate
pip install sacrebleu rouge-score seaborn

# 2. Train Model H
python phase2/train.py --model H

# 3. Train Model L
python phase2/train.py --model L

# 4. Evaluate both models (PPL, BPB, BLEU, chrF, ROUGE-L, diversity)
python phase2/eval_metrics.py --model H --model L

# 5. Attention analysis (heatmaps, entropy, distance)
python phase2/attention_analysis.py --model H
python phase2/attention_analysis.py --model L

# 6. Generate training curves
python phase2/plot_training.py

# 7. Generate PDF report
python phase2/generate_phase2_report.py
```

### Phase 2 Repository Structure

```
phase2/
├── model.py               # Decoder-only Transformer (from scratch)
├── config_H.py            # Model H hyperparameters
├── config_L.py            # Model L hyperparameters
├── train.py               # Training loop with checkpoint resume
├── dataset.py             # DataLoader for tokenized text
├── evaluate.py            # Basic PPL + causal mask verification
├── eval_metrics.py        # Full eval: PPL, BPB, BLEU, chrF, ROUGE-L, diversity
├── attention_analysis.py  # Attention heatmaps, entropy, mean distance
├── plot_training.py       # Training curve plots from JSONL logs
├── generate_phase2_report.py  # PDF report generator
├── logs/
│   ├── train_H.jsonl      # Per-step training log (Model H)
│   └── train_L.jsonl      # Per-step training log (Model L)
├── checkpoints_H/         # Model H checkpoints (Drive link above)
└── checkpoints_L/         # Model L checkpoints (Drive link above)
report/
├── Phase2_Report.pdf
├── training_summary.png
├── val_loss_curve.png
├── val_ppl_curve.png
├── train_loss_curve.png
├── lr_schedule.png
├── attn_Model_H_layer*.png
├── attn_Model_L_layer*.png
├── attn_summary_Model_H.png
└── attn_summary_Model_L.png
```

---

## Phase 1: Dataset Construction & Tokenizer Training

**Phase 1 covers:** corpus collection, preprocessing, train/val/test splits, and BPE tokenizer training.

---

## Language Justification

| Language | Script | Reason for Selection |
|---|---|---|
| **Hindi** | Devanagari | 4th most spoken language globally (~600M speakers). Rich NLP resources available (mC4, Wikipedia, IIT Bombay parallel corpus) making it a well-suited higher-resource target. |
| **Nepali** | Devanagari | ~16M speakers, significantly fewer NLP resources than Hindi despite using the same script — making it a well-motivated lower-resource counterpart. Choosing a same-script language pair allows fair comparison of tokenizer behavior and model capacity effects. |

---

## Dataset Summary

| | Hindi | Nepali |
|---|---|---|
| Total tokens | ~666M | ~1,397M |
| Manual tokens | ~325M (48.8%) | ~1,030M (73.8%) |
| Train lines | 4,623,692 | 2,450,064 |
| Val lines | 121,676 | 64,475 |
| Test lines | 121,677 | 64,476 |
| Tokenizer vocab | 8,000 | 8,000 |
| Fertility (chars/token) | 3.41 | 4.08 |
| UNK rate | 0.00% | 0.00% |

---

## Data Sources

### Hindi
| Source | Type | Tokens |
|---|---|---|
| mC4 (`mc4_lang/hi`) | Automated crawl | ~341M |
| Wikipedia + Wikimedia 20231101.hi | Wiki dump (manual per Q&A) | ~226M |
| cfilt/iitb-english-hindi (IIT Bombay) | Manual parallel corpus | ~68M |
| Helsinki-NLP/opus-100 en-hi | Manual parallel corpus | ~29M |
| Scraped news (BBC Hindi, Prabhat Khabar, Dainik Jagran) | Manual scrape | ~3M |

### Nepali
| Source | Type | Tokens |
|---|---|---|
| mC4 (`mc4_lang/ne`) | Automated crawl | ~367M |
| Wikipedia + Wikimedia 20231101.ne | Wiki dump (manual per Q&A) | ~38M |
| Boredoom17/Nepali-Corpus (IRIIS Nepal) | Manual curated corpus | ~990M |
| Scraped news (nepalnews.com) | Manual scrape | ~2M |

---

## Google Drive Links

> **Dataset files (raw + clean + splits):** (https://drive.google.com/drive/folders/1jK9mWmFgvsXfYRNcSUsFZMswhbnDyawT?usp=sharing)  
> **Tokenizer model files:** https://drive.google.com/drive/folders/1e3VWsGCksFO-zvXRelarV-l7rUyv13zj?usp=sharing
https://drive.google.com/drive/folders/1VWh2DE18KfSic3LpLsSzbTNQodDc-oRy?usp=sharing
> **Phase 1 Report PDF:** https://drive.google.com/file/d/1eXWpL6StjwM1FWerH9iS9vd6rJlV7gYh/view?usp=sharing

The Google Drive folder contains:
```
hindi/
  raw/          ← all raw source files
  clean/        ← preprocessed text files
  splits/       ← train.txt, val.txt, test.txt
  tokenizer/    ← hindi_bpe.model, hindi_bpe.vocab
nepali/
  raw/
  clean/
  splits/
  tokenizer/    ← nepali_bpe.model, nepali_bpe.vocab
```

---

## Directory Structure (this repo)

```
lm_project/
├── hindi/
│   ├── tokenizer/      ← hindi_bpe.model, hindi_bpe.vocab
│   ├── model/          ← (populated in Phase 2)
│   ├── train/          ← (populated in Phase 2)
│   ├── eval/           ← (populated in Phase 2)
│   └── configs/        ← (populated in Phase 2)
├── nepali/
│   ├── tokenizer/      ← nepali_bpe.model, nepali_bpe.vocab
│   ├── model/
│   ├── train/
│   ├── eval/
│   └── configs/
├── report/
│   ├── Phase1_Report.pdf
│   └── report_phase1.md
├── scripts/
│   ├── download_data.py             # mC4 + Wikipedia via HuggingFace
│   ├── download_extra_corpora.py    # Wikimedia sister projects
│   ├── download_opus100_hindi.py    # OPUS-100 en-hi parquet download
│   ├── scrape_hindi_news.py         # BBC Hindi, Prabhat Khabar, Jagran
│   ├── scrape_nepali_targeted.py    # nepalnews.com sitemap scraper
│   ├── preprocess.py                # clean automated sources
│   ├── preprocess_extra.py          # clean manually collected sources
│   ├── split_data.py                # 95/2.5/2.5 train/val/test split
│   ├── train_tokenizer.py           # SentencePiece BPE training
│   └── generate_report_pdf.py       # generates report/Phase1_Report.pdf
└── README.md
```

> **Note:** `data/` is not committed to this repo due to file sizes (raw mC4 files are ~17GB each). All data files are on Google Drive at the link above.

---

## Reproduction Steps

### 1. Environment setup
```bash
python3.12 -m venv venv312
source venv312/bin/activate
pip install datasets==2.19.0 sentencepiece requests pyarrow beautifulsoup4 fpdf2
```

### 2. Download automated sources
```bash
python3 scripts/download_data.py           # mC4 + Wikipedia (hi + ne)
python3 scripts/download_extra_corpora.py  # Wikibooks, Wikiquote, Wiktionary, Wikisource
```

### 3. Download manual parallel corpora
```bash
# IIT Bombay Hindi corpus — download parquet directly:
python3 -c "
import requests, io, pyarrow.parquet as pq
url = 'https://huggingface.co/datasets/cfilt/iitb-english-hindi/resolve/321516f50bdcc1214fa75164c545478976ed84bd/data/train-00000-of-00001.parquet'
r = requests.get(url, stream=True, timeout=120)
data = bytearray()
for chunk in r.iter_content(65536): data.extend(chunk)
t = pq.read_table(io.BytesIO(bytes(data)))
with open('data/hindi/raw/iitb_hindi.txt','w') as f:
    for row in t.to_pydict()['translation']:
        hi = row.get('hi','')
        if hi: f.write(hi + '\n')
print('IIT Bombay done:', len(t), 'rows')
"

# OPUS-100 Hindi
python3 scripts/download_opus100_hindi.py

# Boredoom17 Nepali corpus — download parquet directly:
python3 -c "
import requests, io, pyarrow.parquet as pq
url = 'https://huggingface.co/api/datasets/Boredoom17/Nepali-Corpus/parquet/iriisnepal/train/0.parquet'
r = requests.get(url, stream=True, timeout=300)
data = bytearray()
for chunk in r.iter_content(65536): data.extend(chunk)
t = pq.read_table(io.BytesIO(bytes(data)))
with open('data/nepali/raw/boredoom_nepali.txt','w') as f:
    for row in t.to_pydict()['text']:
        if row: f.write(str(row) + '\n')
print('Boredoom17 done:', len(t), 'rows')
"
```

### 4. Manual news scraping
```bash
python3 scripts/scrape_hindi_news.py       # BBC Hindi, Prabhat Khabar, Dainik Jagran
python3 scripts/scrape_nepali_targeted.py  # nepalnews.com
```

### 5. Preprocess all sources
```bash
python3 scripts/preprocess.py        # mC4, Wikipedia, Wikimedia
python3 scripts/preprocess_extra.py  # scraped news, IIT Bombay, OPUS-100, Boredoom17
```

### 6. Create splits
```bash
python3 scripts/split_data.py
```

### 7. Train tokenizers
```bash
python3 scripts/train_tokenizer.py
```

### 8. Generate report PDF
```bash
python3 scripts/generate_report_pdf.py
```

---

## Preprocessing Pipeline

All text goes through `preprocess.py` / `preprocess_extra.py`:

1. NFC Unicode normalization
2. URL and email removal
3. **Devanagari content filter:** lines with <30% Devanagari characters (U+0900–U+097F) are discarded  
   *(IIT Bombay parallel corpus uses relaxed 10% threshold due to bilingual sentence structure)*
4. Minimum length filter: lines <20 characters discarded
5. Near-deduplication via first-80-chars key
6. Whitespace normalization

Meaningful English proper nouns (e.g. Tata, IRCTC, Google) within Devanagari sentences are retained.

---

## Phase 3: Reasoning Fine-tuning & Attention Analysis

### Phase 3 Results

| | Model H (Hindi) | Model L (Nepali) |
|---|---|---|
| SFT train examples | 2,000 | 2,000 |
| Fine-tuning steps | 500 | 500 |
| Peak LR | 2e-5 | 2e-5 |
| Best val EM (during SFT) | 53.0 % (step 100) | 43.0 % (step 350) |
| Test EM — pretrained | 0.0 % | 0.0 % |
| Test EM — fine-tuned | 37.5 % | 46.5 % |
| Test Entity Acc — fine-tuned | ~62 % | ~68 % |

### Google Drive Links — Phase 3

> **Model H fine-tuned checkpoint (best_ft.pt):** https://drive.google.com/file/d/1AW6q-vOd5tUb_7ZNUjbgKbKoueZ5JRBM/view?usp=sharing
> **Model L fine-tuned checkpoint (best_ft.pt):** https://drive.google.com/file/d/1dyMBT3WVFhGomogtcOqaIT2WkjBBMNKY/view?usp=sharing
> **Phase 3 Report PDF:** (https://drive.google.com/file/d/1J6nmRR_f92sioTuq07XDqr8n0EaB6CmS/view?usp=sharing)

---

## Bonus Ablation: No Positional Embeddings (Model H)

Retrains Model H from scratch with the positional embedding table removed (`InputEmbeddingNoPE`).
All other hyperparameters identical. Parameter reduction: −98,304 (256 × 384).

### Ablation Results Summary

| Metric | With PE | No PE | Δ |
|---|---|---|---|
| Val PPL | 38.86 | 42.25 | +3.39 |
| Test PPL | 38.96 | 42.38 | +3.42 |
| Val BPB | 0.536 | 0.548 | +0.012 |
| Greedy BLEU-4 | 0.93 | 0.60 | −0.33 |
| Avg. attn entropy | 1.407 | 1.498 | +0.091 |
| Avg. attn distance | 4.000 | 3.895 | −0.105 |

### Google Drive Links — Bonus Ablation

> **Ablation checkpoint (no PE, best.pt):** <!-- TODO: upload phase2/ablation/checkpoints/best.pt to Drive and paste link here -->

### Ablation Reproduction Steps

```bash
source venv312/bin/activate

# Train ablation model (Model H, no positional embeddings)
python phase2/ablation/train_nope.py
# Checkpoint: phase2/ablation/checkpoints/best.pt

# Evaluate (PPL/BPB, generation, diversity)
python phase2/ablation/eval_nope.py
# Outputs: report/ablation_eval.json, report/ablation_curves.png

# Attention analysis (heatmaps, entropy, distance)
python phase2/ablation/attention_nope.py
# Outputs: report/ablation_attn_*.png, report/ablation_attn_summary.json

# Generate ablation PDF report
python phase2/ablation/generate_ablation_report.py
# Output: report/Ablation_Report.pdf
```

### Ablation Repository Structure

```
phase2/ablation/
├── model_nope.py          # DecoderLM_NoPE: Model H without positional embedding table
├── config_ablation.py     # Same hyperparams as Model H; separate checkpoint dir
├── train_nope.py          # Full training loop with checkpoint resume
├── eval_nope.py           # PPL/BPB, generation metrics, comparison table, curves plot
├── attention_nope.py      # Side-by-side heatmaps, entropy bars, distance bars
├── generate_ablation_report.py  # PDF report generator
├── checkpoints/           # Ablation checkpoints (Drive link above)
└── logs/
    └── train_nope.jsonl   # Per-step training log
report/
├── Ablation_Report.pdf
├── ablation_curves.png
├── ablation_attn_layer{00,06,11}.png
├── ablation_attn_entropy.png
├── ablation_attn_distance.png
├── ablation_eval.json
└── ablation_attn_summary.json
```

### Phase 3 Reproduction Steps

```bash
# 0. Activate venv (Python 3.12)
source venv312/bin/activate

# 1. Generate synthetic reasoning datasets (Hindi + Nepali)
python3 phase3/generate_hindi_reasoning.py
python3 phase3/generate_nepali_reasoning.py
# Outputs: phase3/data/hindi_reasoning/{train,val,test}.jsonl
#          phase3/data/nepali_reasoning/{train,val,test}.jsonl

# 2. Fine-tune both models on reasoning data
python3 phase3/finetune.py --model H   # ~15-25 min on MPS / CPU
python3 phase3/finetune.py --model L   # ~15-25 min on MPS / CPU
# Saves: phase3/checkpoints_ft_{H,L}/best_ft.pt
# Logs:  phase3/logs/finetune_{H,L}.jsonl

# 3. Evaluate pretrained vs. fine-tuned on the test set
python3 phase3/eval_reasoning.py --model H --model L
# Outputs: report/eval_reasoning_{H,L}.json

# 4. Attention analysis: pretrained vs. fine-tuned heatmaps
python3 phase3/attention_analysis_ft.py --model H
python3 phase3/attention_analysis_ft.py --model L
# Outputs: report/attn_ft_Model_{H,L}_layer*.png
#          report/attn_ft_Model_{H,L}_summary.json

# 5. Generate Phase 3 PDF report
python3 phase3/generate_phase3_report.py
# Output: report/Phase3_Report.pdf
```

### Phase 3 Hyperparameters

| Hyperparameter | Value | Notes |
|---|---|---|
| Learning rate | 2e-5 | Lower than pretraining (1e-3) to preserve representations |
| LR schedule | Linear warmup + cosine decay | 50 warmup steps, decay to 10% of peak |
| Max steps | 500 | Early stopping on val EM |
| Batch size | 32 | Same as pretraining |
| Dropout | 0.05 | Reduced from 0.10 in pretraining |
| Loss masking | ignore_index=-100 | Question tokens ignored; loss only on answer tokens |
| Gradient clip | 1.0 | Same as pretraining |
| Optimizer | AdamW | betas=(0.9, 0.95), eps=1e-8 |

### Phase 3 Repository Structure

```
phase3/
├── generate_hindi_reasoning.py    # Synthetic dataset generator (Hindi)
├── generate_nepali_reasoning.py   # Synthetic dataset generator (Nepali)
├── reasoning_dataset.py           # Dataset + DataLoader with SFT label masking
├── config_ft_H.py                 # Fine-tuning config for Model H
├── config_ft_L.py                 # Fine-tuning config for Model L
├── finetune.py                    # SFT training loop
├── eval_reasoning.py              # Greedy evaluation: EM + entity accuracy
├── attention_analysis_ft.py       # Pretrained vs. fine-tuned attention comparison
├── generate_phase3_report.py      # PDF report generator
├── data/
│   ├── hindi_reasoning/           # train/val/test JSONL (2000/200/200 examples)
│   └── nepali_reasoning/          # train/val/test JSONL (2000/200/200 examples)
├── checkpoints_ft_H/              # Fine-tuned Model H checkpoints (Drive link above)
├── checkpoints_ft_L/              # Fine-tuned Model L checkpoints (Drive link above)
└── logs/
    ├── finetune_H.jsonl           # Per-eval-step training log (Model H)
    └── finetune_L.jsonl           # Per-eval-step training log (Model L)
report/
├── Phase3_Report.pdf              # Final Phase 3 report (this document)
├── eval_reasoning_H.json          # Full evaluation results (Model H)
├── eval_reasoning_L.json          # Full evaluation results (Model L)
├── attn_ft_Model_H_layer*.png     # Attention heatmaps Model H
├── attn_ft_Model_L_layer*.png     # Attention heatmaps Model L
├── attn_ft_Model_H_summary.json   # Entropy summary Model H
└── attn_ft_Model_L_summary.json   # Entropy summary Model L
```
