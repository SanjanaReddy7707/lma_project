"""
Generate Phase 1 PDF report using fpdf2 with Arial Unicode for Devanagari support.
"""
import os
from pathlib import Path
from fpdf import FPDF
from fpdf.enums import XPos, YPos

FONT_PATH = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"
OUT_PATH = "report/Phase1_Report.pdf"

BLUE   = (30,  80, 160)
DBLUE  = (15,  40,  90)
LGRAY  = (245, 245, 250)
MGRAY  = (210, 215, 225)
DGRAY  = (90,  90,  90)
WHITE  = (255, 255, 255)
BLACK  = (20,  20,  20)
GREEN  = (0,  120,  60)


class ReportPDF(FPDF):
    def __init__(self):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.add_font("uni", style="",  fname=FONT_PATH)
        self.add_font("uni", style="B", fname=FONT_PATH)
        self.add_font("uni", style="I", fname=FONT_PATH)
        self.set_margins(20, 25, 20)
        self.set_auto_page_break(auto=True, margin=20)

    # ── header / footer ──────────────────────────────────────────────────────
    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("uni", "B", 8)
        self.set_text_color(*DGRAY)
        self.cell(0, 6, "Phase 1 Report — Dataset Construction & Tokenizer Training", align="L",
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_draw_color(*MGRAY)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(3)

    def footer(self):
        self.set_y(-15)
        self.set_font("uni", "", 8)
        self.set_text_color(*DGRAY)
        self.cell(0, 5, f"Page {self.page_no()}", align="C")

    # ── helpers ───────────────────────────────────────────────────────────────
    def h1(self, text):
        self.ln(3)
        self.set_fill_color(*BLUE)
        self.set_text_color(*WHITE)
        self.set_font("uni", "B", 13)
        self.cell(0, 9, text, fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(3)
        self.set_text_color(*BLACK)

    def h2(self, text):
        self.ln(2)
        self.set_font("uni", "B", 11)
        self.set_text_color(*DBLUE)
        self.cell(0, 7, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_draw_color(*BLUE)
        self.line(self.l_margin, self.get_y(), self.l_margin + 80, self.get_y())
        self.ln(2)
        self.set_text_color(*BLACK)

    def h3(self, text):
        self.ln(1)
        self.set_font("uni", "B", 10)
        self.set_text_color(*DBLUE)
        self.cell(0, 6, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(1)
        self.set_text_color(*BLACK)

    def body(self, text, indent=0):
        self.set_font("uni", "", 10)
        self.set_text_color(*BLACK)
        if indent:
            self.set_x(self.l_margin + indent)
        self.multi_cell(0, 5.5, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(1)

    def bullet(self, text, indent=5):
        self.set_font("uni", "", 10)
        self.set_text_color(*BLACK)
        x = self.l_margin + indent
        self.set_x(x)
        self.cell(4, 5.5, "•")
        self.multi_cell(0, 5.5, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def code_block(self, lines):
        self.set_fill_color(*LGRAY)
        self.set_draw_color(*MGRAY)
        self.set_font("uni", "", 8.5)
        self.set_text_color(50, 50, 50)
        pad = 3
        self.set_x(self.l_margin)
        for ln in lines:
            self.set_x(self.l_margin + pad)
            self.cell(0, 5, ln, fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(2)
        self.set_text_color(*BLACK)

    def kv(self, key, val):
        self.set_font("uni", "B", 10)
        self.set_text_color(*DBLUE)
        self.cell(55, 6, key + ":")
        self.set_font("uni", "", 10)
        self.set_text_color(*BLACK)
        self.multi_cell(0, 6, val, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # ── table ─────────────────────────────────────────────────────────────────
    def table(self, headers, rows, col_widths=None, note=None):
        usable = self.w - self.l_margin - self.r_margin
        if col_widths is None:
            col_widths = [usable / len(headers)] * len(headers)

        # header row
        self.set_fill_color(*DBLUE)
        self.set_text_color(*WHITE)
        self.set_font("uni", "B", 9)
        for h, w in zip(headers, col_widths):
            self.cell(w, 7, h, border=0, fill=True, align="C")
        self.ln()

        # data rows
        self.set_font("uni", "", 9)
        for i, row in enumerate(rows):
            fill = i % 2 == 0
            self.set_fill_color(*(LGRAY if fill else WHITE))
            self.set_text_color(*BLACK)
            is_total = str(row[0]).strip().lower().startswith("total")
            if is_total:
                self.set_font("uni", "B", 9)
                self.set_fill_color(220, 230, 245)
            for cell, w in zip(row, col_widths):
                self.cell(w, 6.5, str(cell), border=0, fill=True)
            self.ln()
            if is_total:
                self.set_font("uni", "", 9)

        self.set_draw_color(*MGRAY)
        self.line(self.l_margin, self.get_y(), self.l_margin + sum(col_widths), self.get_y())
        if note:
            self.set_font("uni", "I", 8.5)
            self.set_text_color(*DGRAY)
            self.ln(1)
            self.multi_cell(0, 5, note, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            self.set_text_color(*BLACK)
        self.ln(3)

    # ── summary box ───────────────────────────────────────────────────────────
    def highlight_box(self, title, lines, color=BLUE):
        self.set_fill_color(235, 242, 255)
        self.set_draw_color(*color)
        x0, y0 = self.get_x(), self.get_y()
        # draw box border later; write content first
        self.set_font("uni", "B", 10)
        self.set_text_color(*color)
        self.cell(0, 6, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_font("uni", "", 10)
        self.set_text_color(*BLACK)
        for ln in lines:
            self.set_x(self.l_margin + 4)
            self.cell(4, 5.5, "•")
            self.multi_cell(0, 5.5, ln, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(2)


# ─────────────────────────────────────────────────────────────────────────────
def build_pdf():
    pdf = ReportPDF()

    # ── COVER PAGE ────────────────────────────────────────────────────────────
    pdf.add_page()
    pdf.set_fill_color(*DBLUE)
    pdf.rect(0, 0, pdf.w, 80, style="F")

    pdf.set_y(18)
    pdf.set_font("uni", "B", 22)
    pdf.set_text_color(*WHITE)
    pdf.cell(0, 12, "Phase 1 Report", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("uni", "B", 15)
    pdf.cell(0, 10, "Dataset Construction & Tokenizer Training", align="C",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("uni", "", 11)
    pdf.set_text_color(200, 215, 240)
    pdf.cell(0, 8, "Language Models and Agents — Course Project", align="C",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.set_y(90)
    pdf.set_text_color(*BLACK)
    pdf.set_font("uni", "B", 11)
    info = [
        ("Languages",    "Hindi (hi) — higher-resource  |  Nepali (ne) — lower-resource"),
        ("Model target", "~25M parameter decoder-only Transformer LM (built from scratch)"),
        ("Tokenizer",    "SentencePiece BPE, 8 000 vocab, trained on train split"),
        ("Submission",   "Phase 1 — Data & Tokenizer"),
    ]
    for k, v in info:
        pdf.set_x(30)
        pdf.set_font("uni", "B", 10)
        pdf.set_text_color(*DBLUE)
        pdf.cell(42, 7, k + ":")
        pdf.set_font("uni", "", 10)
        pdf.set_text_color(*BLACK)
        pdf.multi_cell(0, 7, v, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.ln(8)
    # quick stats box
    pdf.set_fill_color(*LGRAY)
    pdf.set_x(30)
    pdf.set_font("uni", "B", 10)
    pdf.set_text_color(*DBLUE)
    pdf.cell(0, 7, "  Quick Stats", fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    stats = [
        ("Hindi corpus", "~666M tokens  (48.8% manually collected)"),
        ("Nepali corpus", "~1,397M tokens  (73.8% manually collected)"),
        ("Hindi tokenizer", "8 000 vocab | fertility 3.41 chars/tok | 0% UNK"),
        ("Nepali tokenizer", "8 000 vocab | fertility 4.08 chars/tok | 0% UNK"),
    ]
    for k, v in stats:
        pdf.set_x(30)
        pdf.set_font("uni", "B", 10)
        pdf.set_text_color(*BLACK)
        pdf.cell(45, 6.5, "  " + k + ":")
        pdf.set_font("uni", "", 10)
        pdf.cell(0, 6.5, v, fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # ── SECTION 1 — OVERVIEW ──────────────────────────────────────────────────
    pdf.add_page()
    pdf.h1("1.  Dataset Overview")
    pdf.body(
        "Two monolingual Devanagari-script corpora were constructed from scratch for Hindi and Nepali. "
        "Each corpus combines large-scale automated sources (mC4 Common Crawl) with manually collected "
        "material (Wikipedia/Wikimedia dumps, parallel corpora from NLP research institutions, and "
        "news articles scraped with custom Python scripts). Both corpora exceed the 500M-token "
        "minimum and meet the ≥20% manual-collection requirement."
    )

    pdf.h2("1.1  Hindi Corpus")
    hindi_rows = [
        ["mC4  (mc4_lang/hi)",             "Automated crawl", "2,100,247",  "~340.8M"],
        ["Wikipedia  (20231101.hi)",        "Wiki dump",       "988,560",    "~188.2M"],
        ["Wikibooks  (20231101.hi)",        "Wiki dump",       "147,709",    "~13.9M"],
        ["Wikiquote  (20231101.hi)",        "Wiki dump",       "59,366",     "~5.2M"],
        ["Wiktionary  (20231101.hi)",       "Wiki dump",       "161,334",    "~18.7M"],
        ["Wikisource  (20231101.hi)",       "Wiki dump",       "530",        "~0.0M"],
        ["Scraped news (BBC, Prabhat, Jagran)", "Manual scrape", "2,964",   "~2.6M"],
        ["cfilt/iitb-english-hindi  (Hindi side)", "Manual corpus","933,944","~67.7M"],
        ["Helsinki-NLP/opus-100 en-hi  (Hindi side)", "Manual corpus","472,391","~29.2M"],
        ["TOTAL", "", "4,866,045", "~666.3M"],
    ]
    pdf.table(
        ["Source", "Type", "Lines (clean)", "Est. Tokens"],
        hindi_rows,
        col_widths=[82, 32, 30, 26],
        note="Manual fraction (Wikipedia + Wikimedia + scraped news + parallel corpora): ~325.4M / 666.3M = 48.8%"
    )

    pdf.h2("1.2  Nepali Corpus")
    nepali_rows = [
        ["mC4  (mc4_lang/ne)",             "Automated crawl", "2,053,045", "~366.5M"],
        ["Wikipedia  (20231101.ne)",        "Wiki dump",       "156,960",   "~30.0M"],
        ["Wikibooks  (20231101.ne)",        "Wiki dump",       "23,958",    "~2.3M"],
        ["Wiktionary  (20231101.ne)",       "Wiki dump",       "91,833",    "~5.6M"],
        ["Scraped news (nepalnews.com)",    "Manual scrape",   "766",       "~2.2M"],
        ["Boredoom17/Nepali-Corpus  (IRIIS Nepal)", "Manual corpus", "252,453", "~989.9M"],
        ["TOTAL", "", "2,578,015", "~1,396.6M"],
    ]
    pdf.table(
        ["Source", "Type", "Lines (clean)", "Est. Tokens"],
        nepali_rows,
        col_widths=[82, 32, 30, 26],
        note="Manual fraction (Wikipedia + Wikimedia + scraped news + Boredoom17): ~1,030.1M / 1,396.6M = 73.8%"
    )

    # ── SECTION 2 — MANUAL COLLECTION ────────────────────────────────────────
    pdf.add_page()
    pdf.h1("2.  Manual Data Collection")

    pdf.h2("2.1  Hindi — Manual Sources")

    pdf.h3("News Scraping  (scrape_hindi_news.py)")
    pdf.body(
        "Custom scraper targeted three Hindi news publishers: BBC Hindi (bbc.com/hindi), "
        "Prabhat Khabar (prabhatkhabar.com), and Dainik Jagran (jagran.com). "
        "Sitemaps were discovered by fetching /sitemap-index.xml and recursively "
        "resolving sub-sitemap URLs (e.g. news-sitemap.xml). Article body text was extracted "
        "using CSS selectors (.entry-content, .article-body, article p). "
        "~2,413 articles were collected, yielding ~2.6M tokens."
    )

    pdf.h3("IIT Bombay English-Hindi Parallel Corpus  (cfilt/iitb-english-hindi)")
    pdf.body(
        "The CFILT lab at IIT Bombay released a large manually aligned English-Hindi parallel corpus "
        "containing 1,659,083 sentence pairs. The 190MB parquet file was downloaded directly via "
        "the HuggingFace CDN (to avoid streaming connection resets) and the Hindi side was extracted. "
        "Lines with fewer than 10% Devanagari characters were discarded to remove "
        "heavily code-switched or mis-encoded entries. "
        "Retained: 933,944 lines, ~67.7M tokens."
    )

    pdf.h3("OPUS-100  (Helsinki-NLP/opus-100 en-hi)")
    pdf.body(
        "OPUS-100 aggregates 100 languages of parallel text from multiple open corpora "
        "(OpenSubtitles, CCAligned, WikiMatrix, etc.). The en-hi split contains 534,319 "
        "sentence pairs (65MB parquet). Downloaded directly via HuggingFace CDN parquet API. "
        "The Hindi side was extracted and filtered to >=30% Devanagari character ratio. "
        "Retained: 472,391 lines, ~29.2M tokens."
    )

    pdf.h3("Wikipedia & Wikimedia Sister Projects")
    pdf.body(
        "Snapshots of Hindi Wikipedia, Wikibooks, Wikiquote, Wiktionary, and Wikisource "
        "(snapshot 20231101) were downloaded from the wikimedia/wikipedia and related "
        "HuggingFace datasets. Per course Q&A clarification, Wikipedia dumps preprocessed "
        "via script count as manual collection. Wikipedia alone contributes ~188M Hindi tokens."
    )

    pdf.h2("2.2  Nepali — Manual Sources")

    pdf.h3("News Scraping  (scrape_nepali_targeted.py)")
    pdf.body(
        "Custom scraper targeted nepalnews.com. WordPress XML sitemaps were fetched "
        "(13 sitemap files, 25,426 total URLs) and article Devanagari text was extracted "
        "using .entry-content, .post-content, and article CSS selectors. "
        "~766 articles collected, ~2.2M tokens."
    )

    pdf.h3("Boredoom17/Nepali-Corpus  (IRIIS Nepal)")
    pdf.body(
        "A large formal Devanagari Nepali text corpus curated by IRIIS Nepal "
        "(Institute for Research and Innovation in IT), a Nepalese NLP research institution. "
        "Contains 392,982 documents (3.3GB raw) sourced from formal Nepali publications. "
        "Downloaded as parquet directly from HuggingFace CDN. After cleaning and filtering "
        "to >=30% Devanagari ratio, 252,453 documents were retained, yielding ~989.9M tokens. "
        "This single corpus provides the bulk of the Nepali dataset and ensures the manual "
        "fraction far exceeds the 20% requirement."
    )

    pdf.h3("Wikipedia & Wikimedia Sister Projects")
    pdf.body(
        "Snapshots of Nepali Wikipedia, Wikibooks, and Wiktionary (snapshot 20231101) "
        "downloaded and preprocessed via script, contributing ~37.9M tokens to the manual fraction."
    )

    # ── SECTION 3 — PREPROCESSING ─────────────────────────────────────────────
    pdf.add_page()
    pdf.h1("3.  Preprocessing Pipeline")

    pdf.body(
        "All text was processed through a uniform cleaning pipeline implemented in "
        "scripts/preprocess.py (automated sources) and scripts/preprocess_extra.py "
        "(manually collected sources). The same cleaning logic is applied to both."
    )

    steps = [
        "Unicode normalization — NFC via unicodedata.normalize('NFC', text). Ensures consistent "
        "representation of composed Devanagari characters.",
        "URL removal — strips http(s)://... and www.... patterns using regex.",
        "Email removal — strips user@domain patterns.",
        "Devanagari content filter — lines with <30% characters in Unicode range U+0900-U+097F "
        "are discarded. Removes English-dominant or numeric-only lines while preserving "
        "natural code-switching. The IIT Bombay parallel corpus used a relaxed 10% threshold "
        "to accommodate technical English terms in bilingual sentences.",
        "Minimum length filter — lines shorter than 20 characters are discarded.",
        "Near-deduplication — first 80 characters used as a key; duplicate keys are discarded.",
        "Whitespace normalization — multiple spaces and newlines collapsed to single space.",
    ]
    for i, s in enumerate(steps, 1):
        pdf.bullet(f"Step {i}: {s}")

    pdf.ln(2)
    pdf.body(
        "Meaningful English proper nouns (e.g. Tata, IRCTC, Google) that appear naturally "
        "within Hindi/Nepali sentences are retained. Only lines dominated by non-Devanagari "
        "content are removed by the content filter."
    )

    # ── SECTION 4 — SPLITS ────────────────────────────────────────────────────
    pdf.h1("4.  Train / Validation / Test Splits")

    pdf.body(
        "Script: scripts/split_data.py. All clean source files are concatenated into a single "
        "per-language corpus, shuffled with a fixed random seed (seed=42) for reproducibility, "
        "then split in a 95 / 2.5 / 2.5 ratio."
    )

    pdf.h2("4.1  Hindi Splits")
    pdf.table(
        ["Split", "Lines", "Size on Disk"],
        [
            ["train", "4,623,692", "1.86 GB"],
            ["val",   "121,676",   "50.3 MB"],
            ["test",  "121,677",   "49.5 MB"],
            ["TOTAL", "4,867,045", "~1.96 GB"],
        ],
        col_widths=[60, 60, 50],
    )

    pdf.h2("4.2  Nepali Splits")
    pdf.table(
        ["Split", "Lines", "Size on Disk"],
        [
            ["train", "2,450,064", "3.89 GB"],
            ["val",   "64,475",    "106.3 MB"],
            ["test",  "64,476",    "104.2 MB"],
            ["TOTAL", "2,579,015", "~4.1 GB"],
        ],
        col_widths=[60, 60, 50],
    )

    # ── SECTION 5 — TOKENIZER ─────────────────────────────────────────────────
    pdf.add_page()
    pdf.h1("5.  Tokenizer Training & Evaluation")

    pdf.body(
        "Script: scripts/train_tokenizer.py. "
        "Library: SentencePiece (sentencepiece==0.2.0). "
        "Method: Byte Pair Encoding (BPE). "
        "Tokenizers are trained on the train split only and evaluated on the val split."
    )

    pdf.h2("5.1  Training Configuration")
    pdf.table(
        ["Parameter", "Value"],
        [
            ["model_type",             "BPE"],
            ["vocab_size",             "8,000 (both languages)"],
            ["character_coverage",     "0.9995"],
            ["input_sentence_size",    "5,000,000 (random sample from train)"],
            ["Special tokens",         "<pad>=0  <unk>=1  <bos>=2  <eos>=3"],
            ["Trained on",             "train split only (val/test unseen during training)"],
        ],
        col_widths=[80, 90],
    )

    pdf.body(
        "Vocabulary size was set to 8,000 for parameter efficiency within the 25M budget. "
        "A 32K vocab embedding table (32K x 512 x 2 = 33M params) would alone exceed the "
        "entire parameter budget. At 8K, the embedding table uses ~8.2M parameters, leaving "
        "~17M for transformer layers. The 0.00% UNK rate on held-out validation text confirms "
        "that 8K provides complete coverage of both Devanagari scripts -- larger vocab would "
        "add parameters without improving coverage. Course TA guidance also recommended "
        "keeping vocab in the 5-10K range for this model size."
    )

    pdf.h2("5.2  Evaluation Results")
    pdf.table(
        ["Metric", "Hindi", "Nepali"],
        [
            ["Vocabulary size",              "8,000",   "8,000"],
            ["Fertility (tokens/word)",       "1.511",   "1.590"],
            ["Avg chars/token",              "3.41",    "4.08"],
            ["UNK rate (on val split)",      "0.00%",   "0.00%"],
            ["Training data",                "train.txt (4.6M lines)", "train.txt (2.45M lines)"],
        ],
        col_widths=[75, 55, 40],
    )
    pdf.body(
        "Fertility (tokens/word) is the standard measure: a value of 1.0 means no splitting; "
        "values above 1 confirm BPE sub-word tokenization is active. Both languages satisfy "
        "fertility > 1 as required. The chars/token values (3.41 Hindi, 4.08 Nepali) show "
        "each token covers ~3-4 Devanagari characters on average. Zero UNK rate confirms "
        "complete vocabulary coverage of both scripts at 8K."
    )

    pdf.h2("5.3  Tokenization Examples")

    pdf.h3("Hindi (fertility = 3.41 chars/token)")
    pdf.code_block([
        "Input : नमस्ते दुनिया",
        "Output: ['□नम', 'स्ते', '□दुनिया']",
        "",
        "Input : भारत एक विविधताओं से भरा देश है",
        "Output: ['□भारत', '□एक', '□विविध', 'ताओं', '□से', '□भरा', '□देश', '□है']",
        "",
        "Input : आज मौसम बहुत अच्छा है",
        "Output: ['□आज', '□मौसम', '□बहुत', '□अच्छा', '□है']",
    ])

    pdf.h3("Nepali (fertility = 4.08 chars/token)")
    pdf.code_block([
        "Input : नमस्ते दुनिया",
        "Output: ['□नम', 'स्त', 'े', '□दु', 'निया']",
        "",
        "Input : भारत एक विविधताओं से भरा देश है",
        "Output: ['□भारत', '□एक', '□विविध', 'ता', 'ओ', 'ं', '□से', '□भ', 'रा', '□देश', '□है']",
    ])

    pdf.body(
        "The □ prefix (rendered as ’□’ by SentencePiece) marks word boundaries. "
        "Nepali exhibits finer sub-word splits (higher fertility) reflecting its richer "
        "morphology relative to Hindi and the shared 8K vocabulary."
    )

    # ── SECTION 6 — PLANNED ARCHITECTURE ─────────────────────────────────────
    pdf.add_page()
    pdf.h1("6.  Planned Model Architecture (Phase 2)")
    pdf.body(
        "The 8K vocabulary was chosen in coordination with the following planned decoder-only "
        "Transformer architecture to stay within the ~25M parameter budget while allocating "
        "sufficient capacity to FFN layers."
    )
    pdf.table(
        ["Hyperparameter", "Value"],
        [
            ["d_model",              "512"],
            ["n_heads",              "8"],
            ["d_ff  (FFN hidden)",   "2,048  (4x expansion)"],
            ["n_layers",             "6"],
            ["vocab_size",           "8,000"],
            ["max_seq_len",          "512"],
        ],
        col_widths=[90, 80],
    )
    pdf.h2("6.1  Parameter Breakdown")
    pdf.table(
        ["Component", "Parameters"],
        [
            ["Token embedding  (8K x 512)",        "4.1M"],
            ["6 x Self-attention  (4 x 512^2)",     "6.3M"],
            ["6 x FFN  (2 x 512 x 2048)",           "12.6M"],
            ["6 x Layer norms + biases",            "~0.04M"],
            ["TOTAL",                               "~23M"],
        ],
        col_widths=[110, 60],
    )
    pdf.body(
        "At 32K vocab the embedding table alone is 16.4M parameters, leaving only ~8.6M "
        "for all transformer layers (~2-3 layers) -- too shallow for meaningful language "
        "modeling. At 8K, 6 full layers are achievable with a properly sized FFN "
        "(4x expansion ratio), the standard configuration for small language models."
    )

    # ── SECTION 7 — SUMMARY ───────────────────────────────────────────────────
    pdf.add_page()
    pdf.h1("7.  Summary")

    pdf.table(
        ["Metric", "Hindi", "Nepali"],
        [
            ["Total tokens (est.)",          "~666M",    "~1,397M"],
            ["Automated tokens (mC4)",        "~341M  (51.2%)", "~367M  (26.3%)"],
            ["Manual tokens",                 "~325M  (48.8%)", "~1,030M  (73.8%)"],
            ["Meets ≥20% manual requirement", "✓ Yes", "✓ Yes"],
            ["Meets ≥500M token requirement", "✓ Yes", "✓ Yes"],
            ["Train lines",                   "4,623,692",  "2,450,064"],
            ["Val lines",                     "121,676",    "64,475"],
            ["Test lines",                    "121,677",    "64,476"],
            ["Tokenizer vocab size",          "8,000",      "8,000"],
            ["Fertility (chars/token)",        "3.41",       "4.08"],
            ["UNK rate (val split)",           "0.00%",      "0.00%"],
        ],
        col_widths=[85, 50, 35],
    )

    pdf.highlight_box("Key deliverables", [
        "data/hindi/splits/  and  data/nepali/splits/  — train/val/test text files",
        "tokenizers/hindi/hindi_bpe.model + .vocab  (8K BPE)",
        "tokenizers/nepali/nepali_bpe.model + .vocab  (8K BPE)",
        "All raw and clean source files (uploaded to Google Drive)",
        "scripts/  — all collection, preprocessing, splitting, and tokenizer scripts",
    ])

    pdf.h2("6.1  Vocabulary Size vs Fertility Analysis")
    pdf.body(
        "Tokenizers were trained at 4K, 8K, 16K, 24K, and 32K vocabulary sizes and evaluated "
        "on the val split to measure fertility (tokens per word). The table below shows "
        "diminishing returns after 8K -- the gain from 8K to 32K is only ~0.23 tok/word "
        "for Hindi and ~0.29 for Nepali, not worth 4x the embedding table size."
    )
    pdf.table(
        ["Vocab", "Hindi (tok/word)", "Nepali (tok/word)"],
        [
            ["4K",          "1.703", "1.837"],
            ["8K (chosen)", "1.511", "1.590"],
            ["16K",         "1.374", "1.417"],
            ["24K",         "1.317", "1.343"],
            ["32K",         "1.283", "1.299"],
        ],
        col_widths=[60, 60, 50],
    )
    img_path = "report/vocab_fertility.png"
    if Path(img_path).exists():
        pdf.image(img_path, x=pdf.l_margin, w=pdf.w - pdf.l_margin - pdf.r_margin)
        pdf.ln(3)

    pdf.h2("6.2  Note on Vocabulary Overlap")
    pdf.body(
        "The Hindi and Nepali tokenizers share approximately 3,309 tokens (~41% of the 8,000-token "
        "vocabulary). This is expected and does not violate the requirement that vocabularies must "
        "not be shared. Both tokenizers were trained independently on entirely separate corpora "
        "using different train.txt files. The overlap arises naturally because Hindi and Nepali "
        "both use Devanagari script and share many common morphemes, function words, and sub-word "
        "units (e.g. common suffixes like ta, ne, ka). The model weights including separate "
        "embedding tables will be fully independent for each language."
    )

    pdf.h2("6.3  Checklist")
    checks = [
        ("✓", "Hindi corpus ≥ 500M tokens"),
        ("✓", "Nepali corpus ≥ 500M tokens"),
        ("✓", "Hindi manual fraction ≥ 20%  (48.8% achieved)"),
        ("✓", "Nepali manual fraction ≥ 20%  (73.8% achieved)"),
        ("✓", "Train / val / test split at 95 / 2.5 / 2.5"),
        ("✓", "BPE tokenizer trained from scratch on train split"),
        ("✓", "Vocab size 8,000 (within TA-recommended 5–10K range)"),
        ("✓", "Fertility > 1 (sub-word tokenization confirmed)"),
        ("✓", "0% UNK rate on held-out val text"),
        ("✓", "Raw and clean data files available"),
        ("✓", "All collection and preprocessing scripts provided"),
    ]
    for mark, text in checks:
        pdf.set_font("uni", "B", 10)
        pdf.set_text_color(*GREEN)
        pdf.cell(10, 6, mark)
        pdf.set_font("uni", "", 10)
        pdf.set_text_color(*BLACK)
        pdf.cell(0, 6, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # ── OUTPUT ────────────────────────────────────────────────────────────────
    pdf.output(OUT_PATH)
    print(f"PDF saved: {OUT_PATH}")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    build_pdf()
