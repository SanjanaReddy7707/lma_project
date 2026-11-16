"""Build midsem/report.pdf from report.md (+ partB.md, ai_note.md, figures).
Run from lm_project/:  python midsem/make_report.py"""
import os, re
import markdown
from fpdf import FPDF

MID = os.path.dirname(os.path.abspath(__file__))
FONT = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"   # covers Devanagari + Greek/math


def body_html():
    md = open(os.path.join(MID, "report.md"), encoding="utf-8").read()
    md = md.replace("{{PARTB}}", open(os.path.join(MID, "partB.md"), encoding="utf-8").read())
    md = md.replace("{{AINOTE}}", open(os.path.join(MID, "ai_note.md"), encoding="utf-8").read())
    # Python-Markdown only starts a list after a blank line; add one where a list follows a paragraph
    md = re.sub(r"(?m)^(?!\s*-\s|\s*$)(.+)\n(\s*- )", r"\1\n\n\2", md)
    main, _, appendix = md.partition("## Appendix — Figures")
    html = markdown.markdown(main, extensions=["tables"])
    # fpdf's HTML tables need explicit borders/widths to look reasonable
    html = html.replace("<table>", '<table border="1" width="100%">')
    # fpdf cannot render nested tags (<strong>, <code>, <em>) inside table cells: keep their text only
    def flatten(m):
        return m.group(1) + re.sub(r"</?(strong|em|code|b|i)>", "", m.group(2)) + m.group(3)
    html = re.sub(r"(<t[dh][^>]*>)(.*?)(</t[dh]>)", flatten, html, flags=re.S)
    return html, appendix


class PDF(FPDF):
    def footer(self):
        self.set_y(-12); self.set_font("U", size=8)
        self.cell(0, 8, f"{self.page_no()}", align="C")


def main():
    pdf = PDF(format="A4"); pdf.set_margins(15, 14, 15); pdf.set_auto_page_break(True, 14)
    for st in ["", "B", "I", "BI"]:
        pdf.add_font("U", st, FONT)
    pdf.set_text_shaping(True)
    pdf.add_page(); pdf.set_font("U", size=9.5)
    html, appendix = body_html()
    pdf.write_html(html, font_family="U", table_line_separators=True)
    body_pages = pdf.page_no()

    # Appendix: figures (excluded from the page limit)
    pdf.add_page(); pdf.set_font("U", "B", 13); pdf.cell(0, 8, "Appendix — Figures", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("U", size=9.5)
    caption = None
    for line in appendix.splitlines():
        m = re.match(r"\{\{FIG:(.+)\}\}", line.strip())
        if m:
            if caption:
                pdf.multi_cell(0, 5, caption, new_x="LMARGIN", new_y="NEXT")
            pdf.image(os.path.join(MID, "figures", m.group(1)), w=175); pdf.ln(4); caption = None
        elif line.strip():
            caption = line.strip()

    out = os.path.join(MID, "report.pdf")
    pdf.output(out)
    print(f"[saved] {out}  — body pages: {body_pages} (limit 6), total pages: {pdf.page_no()}")


if __name__ == "__main__":
    main()
