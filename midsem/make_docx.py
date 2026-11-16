"""Build midsem/report.docx (research-paper style) from report.md (+ partB.md, ai_note.md, figures).
Run from lm_project/:  python midsem/make_docx.py
Single-column layout chosen on purpose: the Part A/C tables are 5-6 columns wide with
Devanagari text and would be unreadable in a two-column paper layout.
"""
import os, re
from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

MID = os.path.dirname(os.path.abspath(__file__))
FONT = "Arial Unicode MS"          # covers Devanagari + Greek/math on macOS
DEVA = "Kohinoor Devanagari"       # nicer Devanagari fallback if installed

# ---- EDIT THESE TWO LINES BEFORE SUBMITTING -------------------------------
GITHUB_URL = "https://github.com/<your-username>/lma_project   [EDITABLE — paste your repo link here]"
AUTHOR     = "[Your Name] — Roll No. XXXXXXXX3"
# ---------------------------------------------------------------------------

AFFIL = "International Institute of Information Technology, Hyderabad"
COURSE = "CL3410 — Language Models and Agents · Make-up Mid-Semester · Monsoon 2026"
TITLE = "Parameter-Efficient Adaptation and Alignment of a Low-Resource Hindi Language Model: LoRA, DPO, and the Limits of Test-Time Reasoning"


def set_cell_font(cell, bold=False, size=8):
    for p in cell.paragraphs:
        p.paragraph_format.space_after = Pt(1)
        p.paragraph_format.space_before = Pt(1)
        for r in p.runs:
            r.font.name = FONT
            r.font.size = Pt(size)
            r.font.bold = bold
            r._element.rPr.rFonts.set(qn("w:cs"), FONT)


def add_runs(par, text, size=10):
    """Render inline markdown: **bold**, `code`, *italic*."""
    # split on bold/code/italic tokens, keep delimiters
    tokens = re.split(r"(\*\*.+?\*\*|`.+?`|\*.+?\*)", text)
    for tok in tokens:
        if not tok:
            continue
        bold = italic = code = False
        if tok.startswith("**") and tok.endswith("**"):
            tok, bold = tok[2:-2], True
        elif tok.startswith("`") and tok.endswith("`"):
            tok, code = tok[1:-1], True
        elif tok.startswith("*") and tok.endswith("*") and len(tok) > 2:
            tok, italic = tok[1:-1], True
        r = par.add_run(tok)
        r.font.name = "Consolas" if code else FONT
        r.font.size = Pt(size - (0.5 if code else 0))
        r.font.bold = bold
        r.font.italic = italic
        r._element.rPr.rFonts.set(qn("w:cs"), FONT)
        if code:
            r.font.color.rgb = RGBColor(0x55, 0x00, 0x88)


def load_md():
    md = open(os.path.join(MID, "report.md"), encoding="utf-8").read()
    md = md.replace("{{PARTB}}", open(os.path.join(MID, "partB.md"), encoding="utf-8").read())
    md = md.replace("{{AINOTE}}", open(os.path.join(MID, "ai_note.md"), encoding="utf-8").read())
    # drop the raw variant line + the old H1 (we build our own title block)
    md = re.sub(r"^\*\*Variant 3\*\*.*?\n", "", md, count=1)
    md = re.sub(r"^# CL3410 Make-up.*?\n", "", md, count=1, flags=re.M)
    return md


def parse_table(lines, i):
    """lines[i] is a header row '| a | b |'. Return (rows, next_i)."""
    rows = []
    while i < len(lines) and lines[i].strip().startswith("|"):
        cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
        if not re.match(r"^[:\- ]+$", "".join(cells)):   # skip the |---|---| separator
            rows.append(cells)
        i += 1
    return rows, i


def add_table(doc, rows):
    if not rows:
        return
    ncol = max(len(r) for r in rows)
    t = doc.add_table(rows=len(rows), cols=ncol)
    t.style = "Light Grid Accent 1"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = True
    for ri, row in enumerate(rows):
        for ci in range(ncol):
            cell = t.cell(ri, ci)
            txt = row[ci] if ci < len(row) else ""
            # strip inline markdown for cells (docx cells get flat text)
            txt = re.sub(r"\*\*(.+?)\*\*", r"\1", txt)
            txt = re.sub(r"`(.+?)`", r"\1", txt)
            cell.paragraphs[0].text = txt
            set_cell_font(cell, bold=(ri == 0), size=8)
    doc.add_paragraph().paragraph_format.space_after = Pt(4)


def style_heading(doc, text, level):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(10 if level <= 2 else 6)
    p.paragraph_format.space_after = Pt(3)
    r = p.add_run(text)
    r.font.name = FONT
    r.font.bold = True
    r.font.size = Pt({1: 14, 2: 12.5, 3: 11}.get(level, 10.5))
    r.font.color.rgb = RGBColor(0x14, 0x2b, 0x5c)
    r._element.rPr.rFonts.set(qn("w:cs"), FONT)


def main():
    md = load_md()
    main_md, _, appendix = md.partition("## Appendix — Figures")

    doc = Document()
    # base style
    normal = doc.styles["Normal"]
    normal.font.name = FONT
    normal.font.size = Pt(10)
    normal._element.rPr.rFonts.set(qn("w:cs"), FONT)
    for s in doc.sections:
        s.top_margin = Inches(0.7); s.bottom_margin = Inches(0.7)
        s.left_margin = Inches(0.8); s.right_margin = Inches(0.8)

    # ---------------- Title block (research-paper front matter) -------------
    variant = doc.add_paragraph()
    variant.alignment = WD_ALIGN_PARAGRAPH.LEFT
    rv = variant.add_run("Variant 3")
    rv.font.bold = True; rv.font.size = Pt(10); rv.font.name = FONT

    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    rt = title.add_run(TITLE)
    rt.font.bold = True; rt.font.size = Pt(15); rt.font.name = FONT
    rt.font.color.rgb = RGBColor(0x0a, 0x1a, 0x3f)

    for text, size, bold in [(AUTHOR, 11, True), (AFFIL, 10, False), (COURSE, 9, False)]:
        p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(text); r.font.size = Pt(size); r.font.bold = bold; r.font.name = FONT

    # editable GitHub link line
    gh = doc.add_paragraph(); gh.alignment = WD_ALIGN_PARAGRAPH.CENTER
    rg = gh.add_run("Project repository (editable): ")
    rg.font.size = Pt(9.5); rg.font.bold = True; rg.font.name = FONT
    rl = gh.add_run(GITHUB_URL)
    rl.font.size = Pt(9.5); rl.font.name = "Consolas"
    rl.font.color.rgb = RGBColor(0x0b, 0x4d, 0xa0); rl.font.underline = True

    # Abstract
    ab = doc.add_paragraph(); ab.paragraph_format.space_before = Pt(8)
    ra = ab.add_run("Abstract — ")
    ra.font.bold = True; ra.font.italic = True; ra.font.size = Pt(9.5); ra.font.name = FONT
    abstract = ("We adapt a 24.4 M-parameter Hindi transformer pretrained in our course project (Model H) "
                "using two methods written from scratch in PyTorch: Low-Rank Adaptation (LoRA) on the W_Q and "
                "W_V projections of every block, and Direct Preference Optimization (DPO). We verify that the "
                "LoRA-wrapped model is bit-identical to the pretrained model at step 0, sweep rank r in {1,4,16}, "
                "and show that a rank-1 update matches or beats full fine-tuning on three held-out test sets while "
                "using 0.075 % of the parameters and a 330x smaller optimiser state. We build 2,000 in-language "
                "preference pairs and run DPO at beta in {0.1, 0.05}, confirming the step-0 loss of log 2 and "
                "analysing why preference optimisation lowers exact-match accuracy here. Part B presents worked "
                "derivations for LSH deduplication, multilingual sampling, compute-optimal scaling, LoRA memory, "
                "loss masking, and alignment-method selection. Part C studies whether test-time reasoning helps a "
                "1.5 B instruct model on Hindi word problems and whether an LLM judge can be trusted, finding that "
                "longer reasoning fails on template-breaking problems and that an LLM judge exhibits strong "
                "length and position bias (Cohen's kappa = 0 against a symbolic checker).")
    rb = ab.add_run(abstract); rb.font.italic = True; rb.font.size = Pt(9.5); rb.font.name = FONT
    rb._element.rPr.rFonts.set(qn("w:cs"), FONT)

    kw = doc.add_paragraph()
    rk = kw.add_run("Keywords — ")
    rk.font.bold = True; rk.font.italic = True; rk.font.size = Pt(9.5); rk.font.name = FONT
    rk2 = kw.add_run("LoRA; DPO; parameter-efficient fine-tuning; low-resource NLP; Hindi; "
                     "chain-of-thought; self-consistency; LLM-as-judge; scaling laws.")
    rk2.font.italic = True; rk2.font.size = Pt(9.5); rk2.font.name = FONT

    sep = doc.add_paragraph(); sep.paragraph_format.space_after = Pt(6)
    sep.add_run("_" * 95).font.color.rgb = RGBColor(0xbb, 0xbb, 0xbb)

    # ---------------- Body ---------------------------------------------------
    lines = main_md.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        s = line.strip()
        if not s:
            i += 1; continue
        if s.startswith("|"):                     # table
            rows, i = parse_table(lines, i)
            add_table(doc, rows); continue
        m = re.match(r"^(#{1,4})\s+(.*)", s)
        if m:
            style_heading(doc, m.group(2).strip(), len(m.group(1)))
            i += 1; continue
        if s.startswith("- "):                    # bullet
            p = doc.add_paragraph(style="List Bullet")
            p.paragraph_format.space_after = Pt(2)
            add_runs(p, s[2:].strip())
            i += 1; continue
        # plain paragraph (justified)
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.space_after = Pt(5)
        add_runs(p, s)
        i += 1

    # ---------------- Appendix: figures (excluded from 6-page limit) --------
    doc.add_page_break()
    style_heading(doc, "Appendix — Figures", 2)
    caption = None
    for line in appendix.splitlines():
        m = re.match(r"\{\{FIG:(.+)\}\}", line.strip())
        if m:
            path = os.path.join(MID, "figures", m.group(1))
            if os.path.exists(path):
                doc.add_picture(path, width=Inches(6.2))
                doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
            if caption:
                cp = doc.add_paragraph(); cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
                rc = cp.add_run(caption); rc.font.italic = True; rc.font.size = Pt(9); rc.font.name = FONT
                cp.paragraph_format.space_after = Pt(10)
            caption = None
        elif line.strip() and not line.strip().startswith("##"):
            caption = line.strip()

    out = os.path.join(MID, "report.docx")
    doc.save(out)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
