"""Render the project-status Markdown to PDF without a browser.

Headless Chrome and Edge both hang in this sandbox (the Chromium updater
cannot open its named pipe), so the HTML->PDF route is unavailable. This goes
Markdown -> HTML -> fpdf2.write_html instead. Malgun Gothic is registered for
every weight because the document is mostly Korean and fpdf2's built-in fonts
are Latin-1 only.
"""
import argparse
import os
import re

import markdown
from fpdf import FPDF
from fpdf.enums import TextEmphasis
from fpdf.fonts import TextStyle
from fpdf.html import HTML2FPDF

FONTS = r"C:\Windows\Fonts"


class TableSafeHTML(HTML2FPDF):
    """Stop bold leaking out of a paragraph and into the next table.

    On fpdf2 2.9.0 a <p> whose first child is <strong> leaves font_emphasis
    set once the paragraph closes, and the next <table> renders its whole
    body bold. An intervening <pre> happens to clear it, which is why only
    some tables in a document are affected. Each cell starts clean here;
    <th> re-applies its own bold from the tag style right after.
    """

    def handle_starttag(self, tag, attrs):
        if tag in ("td", "th"):
            self.font_emphasis = TextEmphasis.NONE
        super().handle_starttag(tag, attrs)


class Doc(FPDF):
    HTML2FPDF_CLASS = TableSafeHTML

    def __init__(self):
        super().__init__(format="A4", unit="mm")
        self.set_margins(15, 16, 15)
        self.set_auto_page_break(True, margin=18)
        for style, fn in [("", "malgun.ttf"), ("B", "malgunbd.ttf")]:
            self.add_font("malgun", style, os.path.join(FONTS, fn))
        # Malgun has no italic face; reuse the regular so <i> does not crash.
        self.add_font("malgun", "I", os.path.join(FONTS, "malgun.ttf"))
        self.add_font("malgun", "BI", os.path.join(FONTS, "malgunbd.ttf"))
        for style, fn in [("", "consola.ttf"), ("B", "consolab.ttf"),
                          ("I", "consolai.ttf"), ("BI", "consolaz.ttf")]:
            self.add_font("consola", style, os.path.join(FONTS, fn))
        # Code blocks carry Korean comments and Consolas has no Hangul, so
        # those lines would come out blank without a fallback.
        self.set_fallback_fonts(["malgun"])
        self.set_font("malgun", "", 9.5)

    def footer(self):
        self.set_y(-14)
        self.set_font("malgun", "", 7.5)
        self.set_text_color(130, 130, 130)
        self.cell(0, 6, str(self.page_no()), align="C")
        self.set_text_color(0, 0, 0)


# Malgun Gothic has no glyph for either of these, and a missing glyph in a
# results table is worse than an ASCII stand-in: "-0.052" still reads, an
# empty box does not. These run AFTER the Markdown pass, never before: the
# norm bars become "||", and a "||" in a source line turns one pipe-table
# row into a different number of cells, which silently drops the table.
SUBSTITUTIONS = {
    "−": "-",     # MINUS SIGN
    "‖": "||",    # DOUBLE VERTICAL LINE (vector norm)
}


# A fenced block inside a blockquote is not processed by the fenced_code
# extension -- the "> " prefix keeps it out -- so it arrives as an inline
# <code> that happens to contain newlines, and renders as one run-on line.
# Promote those back to real code blocks.
BLOCKQUOTE_FENCE = re.compile(r"<p[^>]*><code>([^<]*\n[^<]*)</code></p>", re.S)

LIST_ITEM = re.compile(r"^\s*(?:[-*+] |\d+\. )")


def preprocess(text):
    """Turn constructs fpdf2's HTML subset renders badly into ones it handles."""
    # Markdown links point at repo-relative paths that mean nothing in a PDF.
    text = re.sub(r"\[([^\]]+)\]\((?!https?:)[^)]+\)", r"\1", text)

    # GitHub starts a list right under a text line; python-markdown wants a
    # blank line first and otherwise folds the items into the paragraph. The
    # checklists in the "next steps" section are written the GitHub way.
    out, in_fence = [], False
    for line in text.split("\n"):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        elif (not in_fence and LIST_ITEM.match(line) and out
                and out[-1].strip() and not LIST_ITEM.match(out[-1])):
            out.append("")
        out.append(line)
    return "\n".join(out)


def unwrap_code_in_cells(html):
    """fpdf2 refuses inline tags inside <td>/<th>, so drop them there.

    Raising instead of degrading is the right call for a layout engine, but a
    table cell naming a config key is exactly where this document uses inline
    code. The cell keeps its text and loses only the monospace styling.
    """
    INLINE = re.compile(r"</?(?:code|strong|em|b|i|a)\b[^>]*>")

    def strip(m):
        return m.group(1) + INLINE.sub("", m.group(2)) + m.group(3)

    return re.sub(r"(<t[dh][^>]*>)(.*?)(</t[dh]>)", strip, html, flags=re.S)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    args = ap.parse_args()

    with open(args.src, encoding="utf-8") as fh:
        text = preprocess(fh.read())

    html = markdown.markdown(
        text, extensions=["tables", "fenced_code", "sane_lists"])
    # fpdf2 draws table rules only when the markup asks for them.
    html = BLOCKQUOTE_FENCE.sub(r"<pre><code>\1</code></pre>", html)
    html = html.replace("<table>", '<table border="1" width="100%">')
    html = unwrap_code_in_cells(html)
    # fpdf2 defaults to single-spaced lines, which is too tight for Hangul.
    # It reads line-height off the style attribute, so set it per block.
    html = re.sub(r"<(p|li|blockquote)>",
                  r'<\1 style="line-height: 1.4">', html)
    for bad, good in SUBSTITUTIONS.items():
        html = html.replace(bad, good)

    pdf = Doc()
    pdf.add_page()
    pdf.write_html(
        html,
        font_family="malgun",
        pre_code_font="consola",
        heading_sizes={"h1": 17, "h2": 12.5, "h3": 10.5},
        tag_styles={
            "h1": TextStyle(color=(20, 22, 26), font_style="B",
                            t_margin=0, b_margin=3),
            "h2": TextStyle(color=(20, 22, 26), font_style="B",
                            t_margin=5, b_margin=2),
            "h3": TextStyle(color=(20, 22, 26), font_style="B",
                            t_margin=3.5, b_margin=1.5),
            "blockquote": TextStyle(color=(52, 58, 64), t_margin=2,
                                    b_margin=2, l_margin=6),
            "li": TextStyle(t_margin=0.6, b_margin=0.6),
        },
        table_line_separators=True,
    )
    pdf.output(args.out)
    print(f"{args.out}  {os.path.getsize(args.out)/1024:.0f} KB  "
          f"{pdf.pages_count} pages")


if __name__ == "__main__":
    main()
