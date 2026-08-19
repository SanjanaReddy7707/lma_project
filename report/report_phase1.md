# Phase 1 Report: Dataset Construction and Tokenizer Training

**Languages:** Hindi (hi) — higher-resource | Nepali (ne) — lower-resource  
**Model size target:** ~25M parameters (decoder-only Transformer)

---

## 1. Data Sources

### 1.1 Hindi (Model H)

| Source | Type | Lines (clean) | Est. Tokens |
|---|---|---|---|
| mC4 (`mc4_lang/hi`) | Automated web crawl | 2,100,247 | ~340.8M |
| Wikipedia (`wikimedia/wikipedia 20231101.hi`) | Wiki dump | 988,560 | ~188.2M |
| Wikibooks (`wikimedia/wikibooks 20231101.hi`) | Wiki dump | 147,709 | ~13.9M |
| Wikiquote (`wikimedia/wikiquote 20231101.hi`) | Wiki dump | 59,366 | ~5.2M |
| Wiktionary (`wikimedia/wiktionary 20231101.hi`) | Wiki dump | 161,334 | ~18.7M |
| Wikisource (`wikimedia/wikisource 20231101.hi`) | Wiki dump | 530 | ~0.0M |
| Scraped news (BBC Hindi, Prabhat Khabar, Dainik Jagran) | Manual scrape | 2,964 | ~2.6M |
| cfilt/iitb-english-hindi (IIT Bombay parallel corpus, Hindi side) | Manual corpus | 933,944 | ~67.7M |
| Helsinki-NLP/opus-100 en-hi (OPUS parallel corpus, Hindi side) | Manual corpus | 472,391 | ~29.2M |
| **Total** | | **4,866,045** | **~666.3M** |

**Manual collection:** Wikipedia dumps + Wikimedia sister projects + scraped news + IIT Bombay corpus + OPUS-100 = **~325.4M tokens (48.8%)** — exceeds the 20% requirement.

---

### 1.2 Nepali (Model L)

| Source | Type | Lines (clean) | Est. Tokens |
|---|---|---|---|
| mC4 (`mc4_lang/ne`) | Automated web crawl | 2,053,045 | ~366.5M |
| Wikipedia (`wikimedia/wikipedia 20231101.ne`) | Wiki dump | 156,960 | ~30.0M |
| Wikibooks (`wikimedia/wikibooks 20231101.ne`) | Wiki dump | 23,958 | ~2.3M |
| Wiktionary (`wikimedia/wiktionary 20231101.ne`) | Wiki dump | 91,833 | ~5.6M |
| Scraped news (nepalnews.com) | Manual scrape | 766 | ~2.2M |
| Boredoom17/Nepali-Corpus (IRIIS Nepal research institution) | Manual corpus | 252,453 | ~989.9M |
| **Total** | | **2,578,015** | **~1,396.6M** |

**Manual collection:** Wikipedia dumps + Wikimedia sister projects + scraped news + Boredoom17 corpus = **~1,030.1M tokens (73.8%)** — exceeds the 20% requirement.

---

## 2. Manual Data Collection Details

### Hindi
- **News scraping:** Custom `scrape_hindi_news.py` script targeted BBC Hindi (`bbc.com/hindi`), Prabhat Khabar (`prabhatkhabar.com`), and Dainik Jagran (`jagran.com`). Collected ~2,413 articles via sitemap discovery (sitemap-index.xml → sub-sitemaps). Extracted article body text using CSS selectors (`.entry-content`, `.article-body`, `article p`).
- **IIT Bombay parallel corpus:** `cfilt/iitb-english-hindi` — 1.66M English–Hindi sentence pairs from IIT Bombay's CFILT lab. Downloaded as parquet directly and extracted the Hindi side. Filtered to lines with ≥10% Devanagari character ratio to exclude heavily code-switched sentences; retained 933,944 lines (~67.7M tokens).
- **OPUS-100:** `Helsinki-NLP/opus-100 en-hi` — 534,319 parallel sentence pairs aggregated from multiple open parallel corpora (OpenSubtitles, CCAligned, WikiMatrix, etc.). The 65 MB parquet file was downloaded directly from the HuggingFace CDN (`huggingface.co/api/datasets/Helsinki-NLP/opus-100/parquet/en-hi/train/0.parquet`) and the Hindi side was extracted via `scripts/preprocess_extra.py`. Filtered to ≥30% Devanagari ratio; retained 472,391 lines (~29.2M tokens).
- **Wikipedia/Wikimedia:** Downloaded using `wikimedia/wikipedia` and `wikimedia/wikibooks` etc. HuggingFace datasets (snapshot `20231101`). Per course Q&A, Wikipedia preprocessed via script counts as manual collection.

### Nepali
- **News scraping:** Custom `scrape_nepali_targeted.py` script targeted nepalnews.com (25,426 URLs from WordPress sitemap across 13 sitemap files). Extracted Nepali article text using `.entry-content`, `.post-content`, `article` CSS selectors.
- **Boredoom17/Nepali-Corpus:** Formally curated Devanagari Nepali text corpus from IRIIS Nepal (Institute for Research and Innovation in IT). Contains 392,982 documents (3.3GB raw) from formal Nepali publications. Downloaded as parquet directly from HuggingFace and filtered with ≥30% Devanagari ratio via `scripts/preprocess_extra.py`; retained 252,453 documents (~989.9M tokens). This corpus was specifically created by a Nepali NLP research institution for language model training.

---

## 3. Preprocessing Pipeline

Script: `scripts/preprocess.py` (standard sources) + `scripts/preprocess_extra.py` (manually collected sources)

Steps applied to all text:
1. **Unicode normalization:** NFC normalization via `unicodedata.normalize('NFC', text)`
2. **URL removal:** Strip `http://...` and `www....` patterns
3. **Email removal:** Strip `user@domain` patterns
4. **Devanagari content filter:** Lines with < 30% Devanagari characters (Unicode range U+0900–U+097F) are discarded — removes English-dominant or numeric-only lines while preserving natural code-switching
   - IIT Bombay parallel corpus used a relaxed 10% threshold since bilingual sentences naturally contain English technical terms
4. **Minimum length filter:** Lines shorter than 20 characters are discarded
5. **Deduplication:** Near-duplicate detection using first 80 characters as key; duplicates discarded
6. **Whitespace normalization:** Collapse multiple spaces/newlines to single space

Non-language gibberish words are discarded by the Devanagari content filter. Meaningful English proper nouns (Tata, IRCTC, Google) embedded in Hindi/Nepali sentences are retained as they appear naturally in the data.

---

## 4. Train / Validation / Test Splits

Script: `scripts/split_data.py`

All clean source files are merged into a single corpus per language, lines are shuffled with a fixed seed (`seed=42`), then split 95/2.5/2.5.

### Hindi

| Split | Lines | Size |
|---|---|---|
| train | 4,623,692 | 1.86 GB |
| val | 121,676 | 50.3 MB |
| test | 121,677 | 49.5 MB |

### Nepali

| Split | Lines | Size |
|---|---|---|
| train | 2,450,064 | 3.89 GB |
| val | 64,475 | 106.3 MB |
| test | 64,476 | 104.2 MB |

---

## 5. Tokenizer Training and Evaluation

Script: `scripts/train_tokenizer.py`  
Library: SentencePiece (`sentencepiece==0.2.0`)  
Method: **BPE (Byte Pair Encoding)**

### Training configuration

| Parameter | Value |
|---|---|
| `model_type` | BPE |
| `vocab_size` | 8,000 (both languages) |
| `character_coverage` | 0.9995 |
| `input_sentence_size` | 5,000,000 (random sample) |
| Special tokens | `<pad>` (0), `<unk>` (1), `<bos>` (2), `<eos>` (3) |

Tokenizers trained on **train split only** (`splits/train.txt`). Evaluated on val split.

**Vocabulary size justification:** The spec recommends "tens of thousands," but 8,000 was chosen for two reasons:

1. **Parameter efficiency:** With a 25M-parameter budget, a 32K vocab embedding table (32K × 512 × 2 = 33M params) would exceed the entire budget before a single transformer layer is added. At 8K, the embedding table uses ~8.2M parameters, leaving ~17M for transformer layers.
2. **Diminishing fertility returns:** The plot below (generated by training tokenizers at 4K, 8K, 16K, 24K, 32K vocab sizes) shows the fertility curve flattens significantly after 8K — the gain from 8K→32K is only ~0.23 tokens/word for Hindi and ~0.29 for Nepali, not worth quadrupling the embedding table.

The 0.00% UNK rate on held-out validation text confirms complete script coverage at 8K.

![Vocab Size vs Fertility](vocab_fertility.png)

| Vocab | Hindi (tok/word) | Nepali (tok/word) |
|---|---|---|
| 4K | 1.703 | 1.837 |
| **8K (chosen)** | **1.511** | **1.590** |
| 16K | 1.374 | 1.417 |
| 24K | 1.317 | 1.343 |
| 32K | 1.283 | 1.299 |

### Evaluation results

| Metric | Hindi | Nepali |
|---|---|---|
| Vocab size | 8,000 | 8,000 |
| **Fertility (tokens/word)** | **1.511** | **1.590** |
| Avg chars/token | 3.41 | 4.08 |
| UNK rate (on val set) | **0.00%** | **0.00%** |

**Fertility > 1 confirmed** for both languages — the tokenizer performs sub-word splitting as required. Fertility is measured as tokens/word on the val split (the standard definition: a value of 1.0 would mean no splitting; values above 1 confirm BPE is active). The chars/token metric (3.41 Hindi, 4.08 Nepali) shows each token covers ~3–4 Devanagari characters on average.

### Tokenization examples (Hindi)

```
'नमस्ते दुनिया'
→ ['▁नम', 'स्ते', '▁दुनिया']

'भारत एक विविधताओं से भरा देश है'
→ ['▁भारत', '▁एक', '▁विविध', 'ताओं', '▁से', '▁भरा', '▁देश', '▁है']

'आज मौसम बहुत अच्छा है'
→ ['▁आज', '▁मौसम', '▁बहुत', '▁अच्छा', '▁है']
```

### Tokenization examples (Nepali)

```
'नमस्ते दुनिया'
→ ['▁नम', 'स्त', 'े', '▁दु', 'निया']

'भारत एक विविधताओं से भरा देश है'
→ ['▁भारत', '▁एक', '▁विविध', 'ता', 'ओ', 'ं', '▁से', '▁भ', 'रा', '▁देश', '▁है']
```

The ▁ prefix marks word boundaries (SentencePiece convention). Nepali achieves finer sub-word splits (4.08 chars/token) reflecting its morphologically richer structure relative to the 8K vocabulary.

---

## 6. Summary

| | Hindi | Nepali |
|---|---|---|
| Total tokens | ~666M | ~1,397M |
| Automated tokens (mC4) | ~341M (51.2%) | ~367M (26.3%) |
| Manual tokens | ~325M (48.8%) | ~1,030M (73.8%) |
| Train lines | 4,623,692 | 2,450,064 |
| Val lines | 121,676 | 64,475 |
| Test lines | 121,677 | 64,476 |
| Tokenizer vocab | 8,000 | 8,000 |
| Fertility (chars/token) | 3.41 | 4.08 |
| UNK rate | 0.00% | 0.00% |

Both corpora exceed the 500M token minimum. Manual collection exceeds 20% for both languages. Tokenizers trained from scratch on domain-matched Devanagari text with zero unknown token rate.

---

## 8. Planned Model Architecture (Phase 2)

The 8K vocabulary was chosen in coordination with the following planned decoder-only Transformer architecture to stay within the ~25M parameter budget while allocating sufficient capacity to FFN layers:

| Hyperparameter | Value |
|---|---|
| `d_model` | 512 |
| `n_heads` | 8 |
| `d_ff` (FFN hidden dim) | 2,048 |
| `n_layers` | 6 |
| `vocab_size` | 8,000 |
| `max_seq_len` | 512 |

**Parameter breakdown:**

| Component | Params |
|---|---|
| Token embedding (8K × 512) | 4.1M |
| 6 × Self-attention (4 × 512²) | 6.3M |
| 6 × FFN (2 × 512 × 2048) | 12.6M |
| 6 × Layer norms + biases | ~0.04M |
| **Total** | **~23M** |

At 32K vocab the embedding table alone would be 16.4M parameters, leaving only ~8.6M for all transformer layers (~2–3 layers) — too shallow for meaningful language modeling. At 8K, 6 full layers are achievable with a properly sized FFN (4× expansion ratio), which is the standard configuration for small language models.

---

## 7. Notes on Vocabulary Overlap

The Hindi and Nepali tokenizers share approximately 3,309 tokens (~41% of the 8,000-token vocabulary). This is expected and does not violate the requirement that vocabularies must not be shared. Both tokenizers were **trained independently** on entirely separate corpora — the overlap arises naturally because Hindi and Nepali both use Devanagari script and share a large proportion of common morphemes, function words, and sub-word units (e.g. common suffixes like `ता`, `ने`, `का`). The model weights — including separate embedding tables — will be fully independent for each language.
