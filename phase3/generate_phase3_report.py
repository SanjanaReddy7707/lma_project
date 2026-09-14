"""
Generate Phase 3 PDF report — Reasoning Fine-tuning, Attention Analysis,
and Final Scientific Comparison.

Run from lm_project/:
    python phase3/generate_phase3_report.py
"""

import os, json, math
from pathlib import Path
from fpdf import FPDF
from fpdf.enums import XPos, YPos

FONT_PATH  = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"
OUT_PATH   = "report/Phase3_Report.pdf"
REPORT_DIR = Path("report")

BLUE   = (30,  80, 160)
DBLUE  = (15,  40,  90)
LGRAY  = (245, 245, 250)
MGRAY  = (210, 215, 225)
DGRAY  = (90,  90,  90)
WHITE  = (255, 255, 255)
BLACK  = (20,  20,  20)
GREEN  = (0,  140,  60)
ORANGE = (200,  80,  10)
RED    = (180,  30,  30)


def load_json(path):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_jsonl(path):
    rows = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


class ReportPDF(FPDF):
    def __init__(self):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.add_font("uni", style="",  fname=FONT_PATH)
        self.add_font("uni", style="B", fname=FONT_PATH)
        self.add_font("uni", style="I", fname=FONT_PATH)
        self.set_margins(20, 25, 20)
        self.set_auto_page_break(auto=True, margin=20)

    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("uni", "B", 8)
        self.set_text_color(*DGRAY)
        self.cell(0, 6, "LMA Project — Phase 3: Reasoning Fine-tuning & Final Report",
                  align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_draw_color(*MGRAY)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(2)

    def footer(self):
        self.set_y(-15)
        self.set_font("uni", "", 8)
        self.set_text_color(*DGRAY)
        self.cell(0, 6, f"Page {self.page_no()}", align="C")

    # ── Layout helpers ────────────────────────────────────────────────────────

    def section_title(self, text, level=1):
        self.ln(4)
        if level == 1:
            self.set_fill_color(*BLUE)
            self.set_text_color(*WHITE)
            self.set_font("uni", "B", 13)
            self.cell(0, 9, f"  {text}", fill=True,
                      new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        else:
            self.set_text_color(*DBLUE)
            self.set_font("uni", "B", 11)
            self.cell(0, 7, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            self.set_draw_color(*BLUE)
            self.line(self.l_margin, self.get_y(),
                      self.l_margin + 60, self.get_y())
        self.set_text_color(*BLACK)
        self.ln(2)

    def body(self, text, size=10):
        self.set_font("uni", "", size)
        self.set_text_color(*BLACK)
        self.multi_cell(0, 6, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(1)

    def kv(self, label, value, label_w=70):
        self.set_font("uni", "B", 9)
        self.set_text_color(*DBLUE)
        self.cell(label_w, 6, label)
        self.set_font("uni", "", 9)
        self.set_text_color(*BLACK)
        self.cell(0, 6, str(value), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def table_header(self, cols, widths, fill=BLUE):
        self.set_fill_color(*fill)
        self.set_text_color(*WHITE)
        self.set_font("uni", "B", 9)
        for col, w in zip(cols, widths):
            self.cell(w, 7, col, border=1, fill=True, align="C")
        self.ln()
        self.set_text_color(*BLACK)

    def table_row(self, vals, widths, shade=False, bold_first=False):
        self.set_fill_color(*LGRAY if shade else WHITE)
        for i, (v, w) in enumerate(zip(vals, widths)):
            style = "B" if (i == 0 and bold_first) else ""
            self.set_font("uni", style, 9)
            self.cell(w, 6, str(v), border=1, fill=shade, align="C" if i > 0 else "L")
        self.ln()

    def notice(self, text, color=ORANGE):
        self.set_fill_color(*LGRAY)
        self.set_text_color(*color)
        self.set_font("uni", "B", 9)
        self.multi_cell(0, 6, f"  {text}", fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_text_color(*BLACK)
        self.ln(1)

    def image_if_exists(self, path, w=170, caption=None):
        if os.path.exists(path):
            self.image(path, x=self.l_margin, w=w)
            if caption:
                self.set_font("uni", "I", 8)
                self.set_text_color(*DGRAY)
                self.cell(0, 5, caption, align="C",
                          new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                self.set_text_color(*BLACK)
            self.ln(2)


# ══════════════════════════════════════════════════════════════════════════════

def build(pdf):

    # ── COVER PAGE ────────────────────────────────────────────────────────────
    pdf.add_page()
    pdf.set_fill_color(*DBLUE)
    pdf.rect(0, 0, 210, 60, "F")
    pdf.set_y(14)
    pdf.set_font("uni", "B", 20)
    pdf.set_text_color(*WHITE)
    pdf.cell(0, 10, "LMA Project — Phase 3", align="C",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("uni", "B", 14)
    pdf.cell(0, 9, "Reasoning Fine-tuning, Attention Analysis", align="C",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(0, 9, "& Final Scientific Comparison", align="C",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_y(65)
    pdf.set_text_color(*BLACK)

    pdf.set_font("uni", "B", 11)
    pdf.set_text_color(*DBLUE)
    pdf.cell(0, 8, "Project Overview", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_text_color(*BLACK)
    pdf.body(
        "This report covers Phase 3 of the LMA project: fine-tuning two independently "
        "trained monolingual Transformer language models (Model H — Hindi; Model L — Nepali) "
        "on synthetic comparative/transitive reasoning tasks, analysing how fine-tuning "
        "reshapes attention patterns, and providing a consolidated cross-phase comparison "
        "of both models."
    )

    # Summary table
    pdf.ln(2)
    pdf.set_font("uni", "B", 10)
    pdf.set_text_color(*DBLUE)
    pdf.cell(0, 7, "Phase 3 at a glance", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_text_color(*BLACK)
    cols   = ["Item", "Model H (Hindi)", "Model L (Nepali)"]
    widths = [80, 50, 40]
    pdf.table_header(cols, widths)
    rows = [
        ("SFT dataset — train", "2 000 examples", "2 000 examples"),
        ("SFT dataset — val/test", "200 / 200", "200 / 200"),
        ("Template families", "8 (T1–T8)", "8 (T1–T8)"),
        ("Leakage prevention", "Held-out entity names", "Held-out entity names"),
        ("Fine-tuning steps", "500", "500"),
        ("Fine-tuning LR", "2 × 10⁻⁵", "2 × 10⁻⁵"),
        ("Pretrained EM (test)", "0.0 %", "0.0 %"),
        ("Fine-tuned EM (test)", "37.5 %", "46.5 %"),
    ]
    for i, r in enumerate(rows):
        pdf.table_row(r, widths, shade=(i % 2 == 0), bold_first=True)

    # ── 1. SYNTHETIC REASONING DATASET ───────────────────────────────────────
    pdf.add_page()
    pdf.section_title("1. Synthetic Reasoning Dataset")
    pdf.body(
        "A bespoke synthetic dataset was generated programmatically for each language "
        "(no existing benchmark used as primary fine-tuning data). Both datasets share "
        "the same eight template families but are rendered entirely in the target "
        "language's script and natural phrasing."
    )

    pdf.section_title("1.1 Template Families", level=2)
    cols   = ["ID", "Description", "Example pattern"]
    widths = [14, 62, 94]
    pdf.table_header(cols, widths)
    tmpl_rows = [
        ("T1", "2-entity, explicit values, ask more/less",
         "X की prop V1 unit है। Y की prop V2 unit है। किसकी prop अधिक है?"),
        ("T2", "2-entity relational (no numbers)",
         "X की prop Y से अधिक है। किसकी prop कम है?"),
        ("T3", "3-entity, explicit values, max/min",
         "X, Y और Z की prop क्रमशः V1, V2 और V3 unit है। सबसे अधिक prop किसकी है?"),
        ("T4", "Chain A>B>C — ask max/min/A-vs-C",
         "X>Y>Z (relational). सबसे कम prop किसकी है?"),
        ("T5", "Chain A<B<C — ask max/min",
         "X<Y<Z (relational). सबसे अधिक prop किसकी है?"),
        ("T6", "Sort 3 entities ascending/descending",
         "... prop के अनुसार बढ़ते क्रम में क्या है?"),
        ("T7", "Equal values — trick question",
         "X और Y दोनों की prop V है। किसकी prop अधिक है?"),
        ("T8", "Item counting (two entities)",
         "X के पास V1 item हैं। Y के पास V2 item हैं।"),
    ]
    for i, r in enumerate(tmpl_rows):
        pdf.table_row(r, widths, shade=(i % 2 == 0), bold_first=True)

    pdf.ln(3)
    pdf.section_title("1.2 Split Statistics", level=2)
    cols   = ["Split", "Hindi examples", "Nepali examples", "Entity sets"]
    widths = [28, 42, 42, 58]
    pdf.table_header(cols, widths)
    split_rows = [
        ("Train", "2 000", "2 000",
         "20 names (राम, श्याम … / सन्तोष, मनिता …)"),
        ("Val",   "200",   "200",
         "6 held-out names"),
        ("Test",  "200",   "200",
         "6 held-out names (different from val)"),
    ]
    for i, r in enumerate(split_rows):
        pdf.table_row(r, widths, shade=(i % 2 == 0), bold_first=True)

    pdf.ln(3)
    pdf.section_title("1.3 Leakage Prevention", level=2)
    pdf.body(
        "All entity names (person names used in comparisons) are strictly partitioned "
        "across splits: train uses 20 names, val uses 6 different names, and test uses "
        "6 more names not seen in either train or val. The comparison templates, "
        "property types, and numerical values are shared across splits — only entity "
        "names are held out. This tests whether the model generalises the comparison "
        "pattern to new referents rather than memorising name-answer associations."
    )

    # ── 2. FINE-TUNING PROTOCOL ───────────────────────────────────────────────
    pdf.add_page()
    pdf.section_title("2. Fine-tuning Protocol")

    pdf.section_title("2.1 Hyperparameters", level=2)
    cols   = ["Hyperparameter", "Model H", "Model L"]
    widths = [80, 45, 45]
    pdf.table_header(cols, widths)
    hparam_rows = [
        ("Pretrained checkpoint", "phase2/checkpoints_H/best.pt (step 6 500)",
         "phase2/checkpoints_L/best.pt (step 11 500)"),
        ("Tokenizer (fixed)", "hindi_bpe.model (8 000 vocab)", "nepali_bpe.model (8 000 vocab)"),
        ("Batch size", "32", "32"),
        ("Gradient accumulation", "1", "1"),
        ("Max fine-tuning steps", "500", "500"),
        ("Warm-up steps", "50", "50"),
        ("Peak learning rate", "2 × 10⁻⁵", "2 × 10⁻⁵"),
        ("LR schedule", "Linear warm-up + cosine decay", "Linear warm-up + cosine decay"),
        ("Weight decay", "0.01", "0.01"),
        ("Gradient clip", "1.0", "1.0"),
        ("Dropout (fine-tuning)", "0.05 (vs 0.10 pretraining)", "0.05 (vs 0.10 pretraining)"),
        ("Loss masking", "Question tokens masked (-100)", "Question tokens masked (-100)"),
        ("Best val EM", "53.0 % (step 100)", "43.0 % (step 350)"),
    ]
    for i, r in enumerate(hparam_rows):
        pdf.table_row(r, widths, shade=(i % 2 == 0), bold_first=True)

    pdf.ln(3)
    pdf.section_title("2.2 Input Format", level=2)
    pdf.body(
        "Each training example is formatted as a causal sequence:\n"
        "    प्रश्न: {question context + question}\n"
        "    उत्तर: {answer}\n\n"
        "A [BOS] token is prepended and [EOS] appended. The cross-entropy loss is "
        "computed only on the answer tokens (question positions receive label -100 "
        "and are ignored). This supervised fine-tuning (SFT) formulation teaches the "
        "model to generate the correct answer given the full question context, while "
        "preserving the pretrained language model on the question distribution."
    )

    # Fine-tuning curves
    pdf.section_title("2.3 Fine-tuning Curves", level=2)

    ft_H = load_jsonl("phase3/logs/finetune_H.jsonl")
    ft_L = load_jsonl("phase3/logs/finetune_L.jsonl")

    if ft_H and ft_L:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax, rows, label, color in [
            (axes[0], ft_H, "Model H (Hindi)", "#1e50a0"),
            (axes[1], ft_L, "Model L (Nepali)", "#a05010"),
        ]:
            steps = [r["step"] for r in rows]
            em    = [r.get("em_acc", 0) for r in rows]
            vloss = [r.get("val_loss", 0) for r in rows]
            ax2   = ax.twinx()
            ax.plot(steps, em,    color=color,   lw=2,   label="EM Accuracy")
            ax2.plot(steps, vloss, color="gray", lw=1.5, linestyle="--", label="Val Loss")
            ax.set_xlabel("Step")
            ax.set_ylabel("EM Accuracy", color=color)
            ax2.set_ylabel("Val Loss",   color="gray")
            ax.set_title(label)
            ax.set_ylim(0, 0.7)
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="lower right")
        plt.tight_layout()
        curve_path = "report/finetune_curves.png"
        plt.savefig(curve_path, dpi=120, bbox_inches="tight")
        plt.close()
        pdf.image_if_exists(curve_path, w=170,
            caption="Fig 1. Fine-tuning curves — EM accuracy (left axis) and val loss (right axis)")

    # ── 3. EVALUATION RESULTS ─────────────────────────────────────────────────
    pdf.add_page()
    pdf.section_title("3. Evaluation Results")

    eval_H = load_json("report/eval_reasoning_H.json")
    eval_L = load_json("report/eval_reasoning_L.json")

    pdf.section_title("3.1 Pretrained vs. Fine-tuned Accuracy (Test Set)", level=2)
    pdf.body(
        "The pretrained models score 0 % exact match on the reasoning test set because "
        "they have not been exposed to the question-answer format and generate fluent "
        "language continuations rather than structured answers. Fine-tuning on 2 000 "
        "examples in the target-format immediately teaches the Q&A format and unlocks "
        "reasoning capability."
    )

    cols   = ["Metric", "Model H pretrained", "Model H fine-tuned",
              "Model L pretrained", "Model L fine-tuned"]
    widths = [38, 30, 30, 30, 32]
    pdf.table_header(cols, widths)

    def pct(v):
        if v is None: return "—"
        return f"{v*100:.1f} %"

    pre_H = eval_H.get("pretrained", {})
    ft_H  = eval_H.get("finetuned",  {})
    pre_L = eval_L.get("pretrained", {})
    ft_L  = eval_L.get("finetuned",  {})

    metric_rows = [
        ("Exact Match (EM)",
         pct(pre_H.get("em_acc")),  pct(ft_H.get("em_acc")),
         pct(pre_L.get("em_acc")),  pct(ft_L.get("em_acc"))),
        ("Entity Accuracy",
         pct(pre_H.get("ent_acc")), pct(ft_H.get("ent_acc")),
         pct(pre_L.get("ent_acc")), pct(ft_L.get("ent_acc"))),
        ("N test examples",
         str(pre_H.get("n", "—")), str(ft_H.get("n", "—")),
         str(pre_L.get("n", "—")), str(ft_L.get("n", "—"))),
    ]
    for i, r in enumerate(metric_rows):
        pdf.table_row(r, widths, shade=(i % 2 == 0), bold_first=True)

    # EM improvement
    imp_H = eval_H.get("improvement", {})
    imp_L = eval_L.get("improvement", {})
    pdf.ln(2)
    pdf.body(
        f"Improvement from fine-tuning — "
        f"Model H: EM +{imp_H.get('em', 0)*100:.1f} pp, "
        f"Entity +{imp_H.get('ent', 0)*100:.1f} pp  |  "
        f"Model L: EM +{imp_L.get('em', 0)*100:.1f} pp, "
        f"Entity +{imp_L.get('ent', 0)*100:.1f} pp"
    )

    # Per-template breakdown
    pdf.section_title("3.2 Per-Template Accuracy (Fine-tuned models)", level=2)
    cols   = ["Template", "H EM", "H Ent", "L EM", "L Ent", "Description"]
    widths = [22, 16, 16, 16, 16, 84]
    pdf.table_header(cols, widths)

    tmpl_labels = {
        "T1": "2-entity, values",
        "T2": "2-entity, relational",
        "T3": "3-entity, values max/min",
        "T4": "Chain A>B>C",
        "T5": "Chain A<B<C",
        "T6": "Sort 3 entities",
        "T7": "Equal values",
        "T8": "Item counting",
    }
    ft_H_tmpl = ft_H.get("by_template", {})
    ft_L_tmpl = ft_L.get("by_template", {})

    for i, tmpl in enumerate(sorted(tmpl_labels)):
        dH = ft_H_tmpl.get(tmpl, {})
        dL = ft_L_tmpl.get(tmpl, {})
        row = (
            tmpl,
            pct(dH.get("em_acc")),
            pct(dH.get("ent_acc")),
            pct(dL.get("em_acc")),
            pct(dL.get("ent_acc")),
            tmpl_labels[tmpl],
        )
        pdf.table_row(row, widths, shade=(i % 2 == 0), bold_first=True)

    # ── 4. QUALITATIVE ANALYSIS ───────────────────────────────────────────────
    pdf.add_page()
    pdf.section_title("4. Qualitative Analysis — Successes and Failures")

    for model_key, eval_data, label in [
        ("H", eval_H, "Model H (Hindi)"),
        ("L", eval_L, "Model L (Nepali)"),
    ]:
        pdf.section_title(f"4.{1 if model_key=='H' else 2} {label}", level=2)
        samples = eval_data.get("finetuned", {}).get("samples", [])
        if not samples:
            pdf.body("No samples available.")
            continue

        successes = [s for s in samples if s.get("em")][:3]
        failures  = [s for s in samples if not s.get("em")][:3]

        for grp, grp_label in [(successes, "Successes"), (failures, "Failures")]:
            if not grp:
                continue
            pdf.set_font("uni", "B", 9)
            color = GREEN if grp_label == "Successes" else RED
            pdf.set_text_color(*color)
            pdf.cell(0, 7, f"  {grp_label}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_text_color(*BLACK)
            for ex in grp:
                pdf.set_fill_color(*LGRAY)
                pdf.set_font("uni", "", 8)
                inp_short = ex.get("input", "")[:120]
                pdf.set_x(pdf.l_margin)
                pdf.multi_cell(0, 5, f"  Q: {inp_short}", fill=True,
                               new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                pdf.set_font("uni", "B", 8)
                pdf.set_text_color(*GREEN if grp_label == "Successes" else RED)
                pdf.set_x(pdf.l_margin)
                pdf.multi_cell(0, 5, f"  Pred: {ex.get('pred','')[:80]}",
                               new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                pdf.set_text_color(*DGRAY)
                pdf.set_x(pdf.l_margin)
                pdf.multi_cell(0, 5, f"  Gold: {ex.get('gold','')[:80]}",
                               new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                pdf.set_text_color(*BLACK)
                pdf.ln(1)

    # Error pattern summary
    pdf.section_title("4.3 Error Patterns", level=2)
    pdf.body(
        "Common failure modes across both models:\n"
        "  1. Wrong entity selection on T4/T5 chains: the model occasionally picks "
        "the middle entity (B) instead of the extreme (A or C) in transitive chains, "
        "suggesting multi-hop inference is harder than direct comparison.\n"
        "  2. T6 ordering errors: the model sometimes reverses the order symbol "
        "(prints > instead of <) while correctly identifying the entities, indicating "
        "format confusion rather than reasoning failure.\n"
        "  3. T7 equal-value trick: the model occasionally picks one entity instead "
        "of answering 'both are equal', suggesting it defaults to the comparison "
        "pattern it saw most often during fine-tuning.\n"
        "  4. Partial entity match: some generated answers contain the correct entity "
        "but with different phrasing (explains Entity Accuracy > EM)."
    )

    # ── 5. ATTENTION ANALYSIS ─────────────────────────────────────────────────
    pdf.add_page()
    pdf.section_title("5. Attention Analysis: Pretrained vs. Fine-tuned")

    attn_H = load_json("report/attn_ft_Model_H_summary.json")
    attn_L = load_json("report/attn_ft_Model_L_summary.json")

    pdf.body(
        "Attention entropy (H = -sum p log p over key positions) measures how diffuse "
        "each attention head is.  Lower entropy = sharper, more selective attention. "
        "Entropy was computed for pretrained and fine-tuned models on two reasoning "
        "prompts per language: one direct-comparison (T1) and one transitive chain (T4)."
    )

    # Entropy comparison table
    pdf.section_title("5.1 Entropy Change by Layer", level=2)
    cols   = ["Model", "Layer", "Layer type",
              "Pre entropy", "FT entropy", "Δ entropy"]
    widths = [20, 16, 26, 28, 28, 32]
    pdf.table_header(cols, widths)

    rows_attn = []
    for model_name, attn_data, layers in [
        ("Model H", attn_H, attn_H.get("layer_analysed", [0, 6, 11])),
        ("Model L", attn_L, attn_L.get("layer_analysed", [0, 3, 5])),
    ]:
        examples = attn_data.get("examples", [])
        if not examples:
            continue
        ex = examples[0]   # use first example
        layer_labels = ["Early", "Middle", "Late"]
        for layer, lbl in zip(layers, layer_labels):
            key = f"layer{layer}"
            pre = ex.get("pre_entropy", {}).get(key, 0)
            ft  = ex.get("ft_entropy",  {}).get(key, 0)
            delta = ft - pre
            rows_attn.append((model_name, f"Layer {layer}", lbl,
                              f"{pre:.4f}", f"{ft:.4f}",
                              f"{delta:+.4f}"))

    for i, r in enumerate(rows_attn):
        pdf.table_row(r, widths, shade=(i % 2 == 0), bold_first=True)

    pdf.ln(3)
    pdf.body(
        "Key observations:\n"
        "  Early layers (Layer 0): entropy is nearly identical between pretrained and "
        "fine-tuned models for both H and L. General Devanagari language representations "
        "are preserved by fine-tuning.\n"
        "  Late layers: the final layer shows the largest entropy drop — Layer 11 of "
        "Model H drops by ~0.40 nats on the direct-comparison example, Layer 5 of "
        "Model L drops by ~0.24 nats. Sharper late-layer attention indicates the model "
        "learns to selectively focus on the relevant entity tokens and comparison "
        "values when generating the answer.\n"
        "  Middle layers show moderate sharpening, consistent with the view that "
        "fine-tuning gradually reroutes information from lower representation layers "
        "toward increasingly task-specific upper layers."
    )

    # Attention heatmaps
    pdf.section_title("5.2 Attention Heatmaps", level=2)
    for fname, cap in [
        ("report/attn_ft_Model_H_layer00.png", "Model H — Early layer (Layer 0): pretrained vs. fine-tuned"),
        ("report/attn_ft_Model_H_layer11.png", "Model H — Late layer (Layer 11): pretrained vs. fine-tuned"),
        ("report/attn_ft_Model_L_layer00.png", "Model L — Early layer (Layer 0): pretrained vs. fine-tuned"),
        ("report/attn_ft_Model_L_layer05.png", "Model L — Late layer (Layer 5): pretrained vs. fine-tuned"),
    ]:
        pdf.image_if_exists(fname, w=170, caption=cap)

    # ── 6. FINAL COMPARISON ───────────────────────────────────────────────────
    pdf.add_page()
    pdf.section_title("6. Final Cross-Phase Comparison")

    pdf.section_title("6.1 Data Scale and Quality", level=2)
    eval_model_H = load_json("report/eval_model_H.json")
    eval_model_L = load_json("report/eval_model_L.json")

    cols   = ["Dimension", "Model H (Hindi)", "Model L (Nepali)"]
    widths = [72, 54, 44]
    pdf.table_header(cols, widths)
    data_rows = [
        ("Language", "Hindi (hi)", "Nepali (ne)"),
        ("Script", "Devanagari", "Devanagari"),
        ("Total corpus tokens", "~666 M", "~1 397 M"),
        ("Train lines", "4 623 692", "2 450 064"),
        ("Primary automated source", "mC4 (341 M tok)", "mC4 (367 M tok)"),
        ("Primary manual source", "Wikipedia/IITB (294 M)", "Boredoom17 (990 M)"),
        ("Tokenizer fertility (chars/tok)", "3.41", "4.08"),
        ("UNK rate", "0.00 %", "0.00 %"),
    ]
    for i, r in enumerate(data_rows):
        pdf.table_row(r, widths, shade=(i % 2 == 0), bold_first=True)

    pdf.ln(3)
    pdf.section_title("6.2 Language Modelling Results", level=2)
    cols   = ["Metric", "Model H (Hindi)", "Model L (Nepali)"]
    widths = [72, 54, 44]
    pdf.table_header(cols, widths)

    ppl_H = eval_model_H.get("ppl_bpb", {})
    ppl_L = eval_model_L.get("ppl_bpb", {})
    lm_rows = [
        ("Architecture", "12L × d384 × 6H", "6L × d512 × 8H"),
        ("Parameters", "24.4 M", "23.1 M"),
        ("Training steps", "6 500", "12 000"),
        ("Val loss", f"{ppl_H.get('val',{}).get('loss', '—'):.4f}", f"{ppl_L.get('val',{}).get('loss', '—'):.4f}"),
        ("Val PPL", f"{ppl_H.get('val',{}).get('ppl', '—'):.2f}", f"{ppl_L.get('val',{}).get('ppl', '—'):.2f}"),
        ("Val BPB", f"{ppl_H.get('val',{}).get('bpb', '—'):.4f}", f"{ppl_L.get('val',{}).get('bpb', '—'):.4f}"),
        ("Test PPL", f"{ppl_H.get('test',{}).get('ppl', '—'):.2f}", f"{ppl_L.get('test',{}).get('ppl', '—'):.2f}"),
        ("Test BPB", f"{ppl_H.get('test',{}).get('bpb', '—'):.4f}", f"{ppl_L.get('test',{}).get('bpb', '—'):.4f}"),
    ]
    for i, r in enumerate(lm_rows):
        pdf.table_row(r, widths, shade=(i % 2 == 0), bold_first=True)

    pdf.ln(3)
    pdf.section_title("6.3 Reasoning Results (Fine-tuned)", level=2)
    cols   = ["Metric", "Model H (Hindi)", "Model L (Nepali)"]
    widths = [72, 54, 44]
    pdf.table_header(cols, widths)
    r_rows = [
        ("SFT train examples", "2 000", "2 000"),
        ("Fine-tuning steps", "500", "500"),
        ("Best val EM (during SFT)", "53.0 % (step 100)", "43.0 % (step 350)"),
        ("Test EM — pretrained", "0.0 %", "0.0 %"),
        ("Test EM — fine-tuned", f"{ft_H.get('em_acc',0)*100:.1f} %", f"{ft_L.get('em_acc',0)*100:.1f} %"),
        ("Test Entity Acc — fine-tuned", f"{ft_H.get('ent_acc',0)*100:.1f} %", f"{ft_L.get('ent_acc',0)*100:.1f} %"),
    ]
    for i, r in enumerate(r_rows):
        pdf.table_row(r, widths, shade=(i % 2 == 0), bold_first=True)

    # ── 7. ANSWERS TO GUIDING QUESTIONS ──────────────────────────────────────
    pdf.add_page()
    pdf.section_title("7. Answers to Final Report Guiding Questions")

    qa_pairs = [
        ("Q1. How did data scale and quality differ between Model H and Model L?",
         "Model H (Hindi) trained on ~666 M tokens from diverse, well-curated sources "
         "(mC4 + Wikipedia + IITB parallel corpus + OPUS-100), with moderate fertility "
         "(3.41 chars/token) indicating clean BPE coverage of Hindi morphology. "
         "Model L (Nepali) had a larger raw corpus (~1 397 M tokens) but it was "
         "dominated by a single curated source (Boredoom17, 73.8 % of tokens), which "
         "is encyclopedic/literary in style. Nepali's higher fertility (4.08) reflects "
         "its more agglutinative morphology relative to Hindi, requiring more tokens "
         "to represent the same surface text."),
        ("Q2. How do LM and reasoning results compare across the two resource tiers?",
         "Despite Nepali's nominally larger corpus, Model L achieves higher perplexity "
         "(PPL 45.6 vs 38.9 for Hindi) — primarily because its training corpus is less "
         "diverse (one dominant source) and the model is shallower (6 vs 12 layers). "
         "On reasoning fine-tuning, Model L outperforms Model H on the test set "
         "(46.5 % vs 37.5 % EM), which is somewhat surprising. This is explained by "
         "Model L's wider hidden dimension (d=512 vs d=384) providing more per-layer "
         "capacity, and the simpler 6-layer architecture converging more stably on the "
         "small 2 000-example SFT set."),
        ("Q3. What tokenizer/corpus factors most affected the lower-resource model?",
         "The higher fertility of the Nepali BPE tokenizer (4.08 vs 3.41) means each "
         "Nepali sentence spans more tokens, increasing effective sequence length and "
         "making within-context_len packing less efficient. The corpus's stylistic "
         "narrowness (literary Nepali dominant) may also limit the model's exposure to "
         "colloquial comparison phrasing, which partly explains why the reasoning "
         "fine-tuning curves for Model L take longer to peak (step 350 vs step 100 "
         "for Model H)."),
        ("Q4. What evidence explains the observed differences?",
         "1. BPB (bits-per-byte) normalises for fertility: Model L's BPB (0.466) is "
         "actually lower than Model H's (0.540), meaning Model L compresses its "
         "language more efficiently per byte — consistent with its simpler corpus "
         "structure. 2. Attention entropy analysis shows fine-tuning causes larger "
         "absolute entropy drops in Model H's final layers (Δ ≈ -0.40) than in Model L "
         "(Δ ≈ -0.24), reflecting that Model H's deeper stack has more room for "
         "task-specific specialisation. 3. Greedy generation diversity (Distinct-1/2 "
         "from Phase 2) was higher for Model L at temperature 1.0, suggesting it "
         "encodes a more varied Nepali vocabulary distribution that fine-tuning can "
         "exploit for structured answers."),
    ]

    for q, a in qa_pairs:
        pdf.set_font("uni", "B", 9)
        pdf.set_text_color(*DBLUE)
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(0, 6, q, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_text_color(*BLACK)
        pdf.set_font("uni", "", 9)
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(0, 5.5, a, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(3)

    # ── 8. SUBMISSION CHECKLIST ───────────────────────────────────────────────
    pdf.add_page()
    pdf.section_title("8. Submission Checklist")

    checklist = [
        ("Dataset & preprocessing code", "scripts/ — download_data.py, preprocess.py, split_data.py, train_tokenizer.py", True),
        ("Separate tokenizers & vocabularies", "hindi/tokenizer/hindi_bpe.{model,vocab}; nepali/tokenizer/nepali_bpe.{model,vocab}", True),
        ("Transformer implementation", "phase2/model.py — from-scratch: embedding, causal self-attn, FFN, LN, residual blocks", True),
        ("Pretrained checkpoints (Drive)", "README — Model H & L Google Drive links (phase2/checkpoints_{H,L}/best.pt)", True),
        ("Finetuned checkpoints (Drive)", "README — Model H & L fine-tuned Drive links (upload phase3/checkpoints_ft_{H,L}/best_ft.pt)", False),
        ("Training logs", "phase2/logs/train_{H,L}.jsonl; phase3/logs/finetune_{H,L}.jsonl", True),
        ("Evaluation scripts & results", "phase2/eval_metrics.py; phase3/eval_reasoning.py; report/eval_*.json", True),
        ("Attention visualisations", "report/attn_*.png (Phase 2 heatmaps + Phase 3 pre/post comparison)", True),
        ("Final report", "report/Phase3_Report.pdf (this document)", True),
        ("README with reproduction steps", "README.md — Phase 1/2/3 steps, all Drive links", False),
    ]

    cols   = ["Item", "Location", "Status"]
    widths = [64, 96, 10]
    pdf.table_header(cols, widths)
    for i, (item, loc, done) in enumerate(checklist):
        pdf.set_fill_color(*(LGRAY if i % 2 == 0 else WHITE))
        pdf.set_font("uni", "B", 8)
        pdf.cell(64, 6, item, border=1, fill=(i%2==0))
        pdf.set_font("uni", "", 8)
        pdf.cell(96, 6, loc, border=1, fill=(i%2==0))
        symbol = "OK" if done else "TODO"
        pdf.set_font("uni", "B", 8)
        pdf.set_text_color(*GREEN if done else ORANGE)
        pdf.cell(10, 6, symbol, border=1, fill=(i%2==0), align="C")
        pdf.set_text_color(*BLACK)
        pdf.ln()

    pdf.ln(3)
    pdf.notice(
        "TODO items: (1) Upload phase3/checkpoints_ft_H/best_ft.pt and "
        "phase3/checkpoints_ft_L/best_ft.pt to Google Drive and add links to README. "
        "(2) Add Phase 3 reproduction steps to README.",
        color=ORANGE,
    )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os, sys
    os.chdir(os.path.join(os.path.dirname(__file__), ".."))
    pdf = ReportPDF()
    build(pdf)
    os.makedirs("report", exist_ok=True)
    pdf.output(OUT_PATH)
    print(f"[saved] {OUT_PATH}")
