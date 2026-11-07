"""Render the project-status Markdown to .docx.

Neither pandoc nor the docx npm package is available here, so this walks the
HTML that python-markdown produces and builds the document with python-docx.
Doing it by hand buys one thing the PDF build could not have: python-docx
accepts runs inside table cells, so bold numbers and monospaced config keys
survive in the tables instead of flattening to plain text.
"""
import argparse
import os
import re
import xml.etree.ElementTree as ET

import markdown
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor

BODY_FONT = "Malgun Gothic"
CODE_FONT = "Consolas"
INK = RGBColor(0x16, 0x18, 0x1D)
MUTED = RGBColor(0x45, 0x4B, 0x54)

# A fenced block inside a blockquote is not processed by the fenced_code
# extension -- the "> " prefix keeps it out -- so it arrives as an inline
# <code> that happens to contain newlines, and renders as one run-on line.
# Promote those back to real code blocks.
BLOCKQUOTE_FENCE = re.compile(r"<p[^>]*><code>([^<]*\n[^<]*)</code></p>", re.S)

LIST_ITEM = re.compile(r"^\s*(?:[-*+] |\d+\. )")


def preprocess(text):
    """Match GitHub on the two things python-markdown reads differently."""
    # Repo-relative links mean nothing in a Word file; keep the label.
    text = re.sub(r"\[([^\]]+)\]\((?!https?:)[^)]+\)", r"\1", text)

    # GitHub starts a list directly under a text line; python-markdown needs a
    # blank line first and otherwise folds the items into the paragraph.
    out, in_fence = [], False
    for line in text.split("\n"):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        elif (not in_fence and LIST_ITEM.match(line) and out
                and out[-1].strip() and not LIST_ITEM.match(out[-1])):
            out.append("")
        out.append(line)
    return "\n".join(out)


def set_font(run, name):
    """Apply a font to Latin and East Asian text alike.

    Setting run.font.name alone leaves Hangul to Word's default theme font,
    so the Korean and the Latin halves of a sentence render in two faces.
    """
    run.font.name = name
    run._element.rPr.rFonts.set(qn("w:eastAsia"), name)


def shade(element, fill):
    # python-docx elements are lxml, so they need OxmlElement, not ET.
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:fill"), fill)
    element.append(shd)


class Builder:
    def __init__(self, doc):
        self.doc = doc

    # ---- inline ----------------------------------------------------------

    def runs(self, para, node, bold=False, italic=False, mono=False):
        """Emit node's text and children as runs, carrying emphasis down."""
        if node.text:
            self.run(para, node.text, bold, italic, mono)
        for child in node:
            tag = child.tag.lower()
            self.runs(
                para, child,
                bold=bold or tag in ("strong", "b"),
                italic=italic or tag in ("em", "i"),
                mono=mono or tag == "code",
            )
            if child.tail:
                self.run(para, child.tail, bold, italic, mono)

    def run(self, para, text, bold, italic, mono):
        # Markdown wraps a paragraph over several source lines; those newlines
        # are not breaks, and a "\n" inside a run is not one in Word either.
        text = re.sub(r"\s*\n\s*", " ", text)
        if not text:
            return
        r = para.add_run(text)
        r.bold, r.italic = bold, italic
        set_font(r, CODE_FONT if mono else BODY_FONT)
        r.font.size = Pt(8.5 if mono else 10)
        r.font.color.rgb = INK

    # ---- blocks ----------------------------------------------------------

    def heading(self, node, level):
        p = self.doc.add_heading(level=level)
        self.runs(p, node, bold=True)
        for r in p.runs:
            r.font.color.rgb = INK
            r.font.size = Pt({1: 18, 2: 14, 3: 11.5}[level])

    def paragraph(self, node, style=None):
        p = self.doc.add_paragraph(style=style)
        self.runs(p, node)
        return p

    def quote(self, node):
        for child in node:
            if child.tag == "pre":
                self.code(child, indent=True)
                continue
            # A quote can hold a list, and passing the <ul> itself to
            # paragraph() would flatten the items into one run of text.
            items = ([child] if child.tag not in ("ul", "ol")
                     else child.findall("li"))
            for item in items:
                p = self.doc.add_paragraph()
                if item.tag == "li":
                    self.run(p, "•  ", False, False, False)
                self.runs(p, item)
                fmt = p.paragraph_format
                fmt.left_indent = Pt(30 if item.tag == "li" else 18)
                fmt.space_before = Pt(3)
                for r in p.runs:
                    if r.font.color.rgb == INK:
                        r.font.color.rgb = MUTED
                shade(p._p.get_or_add_pPr(), "F4F6F8")

    def code(self, node, indent=False):
        text = "".join(node.itertext()).rstrip("\n")
        for line in text.split("\n"):
            # One paragraph per line: a \n inside a run is not a line break.
            p = self.doc.add_paragraph()
            fmt = p.paragraph_format
            fmt.space_before = fmt.space_after = Pt(0)
            fmt.left_indent = Pt(24 if indent else 12)
            r = p.add_run(line or " ")
            set_font(r, CODE_FONT)
            r.font.size = Pt(8.5)
            shade(p._p.get_or_add_pPr(), "F4F6F8")
        self.doc.add_paragraph().paragraph_format.space_after = Pt(4)

    def listing(self, node, ordered):
        style = "List Number" if ordered else "List Bullet"
        for li in node.findall("li"):
            p = self.doc.add_paragraph(style=style)
            self.runs(p, li)

    def table(self, node):
        rows = [r for r in node.iter("tr")]
        cells = [[c for c in r if c.tag in ("td", "th")] for r in rows]
        t = self.doc.add_table(rows=len(rows), cols=max(len(c) for c in cells))
        t.style = "Table Grid"
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        for ri, row in enumerate(cells):
            for ci, cell in enumerate(row):
                tc = t.cell(ri, ci)
                p = tc.paragraphs[0]
                fmt = p.paragraph_format
                fmt.space_before = fmt.space_after = Pt(1)
                fmt.line_spacing = 1.0
                header = cell.tag == "th"
                self.runs(p, cell, bold=header)
                for r in p.runs:
                    r.font.size = Pt(8.5)
                if header:
                    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    shade(tc._tc.get_or_add_tcPr(), "E9ECEF")
        self.doc.add_paragraph().paragraph_format.space_after = Pt(4)

    def rule(self):
        p = self.doc.add_paragraph()
        pbdr = OxmlElement("w:pBdr")
        p._p.get_or_add_pPr().append(pbdr)
        bottom = OxmlElement("w:bottom")
        pbdr.append(bottom)
        bottom.set(qn("w:val"), "single")
        bottom.set(qn("w:sz"), "6")
        bottom.set(qn("w:color"), "CED4DA")

    def walk(self, root):
        for node in root:
            tag = node.tag.lower()
            if tag in ("h1", "h2", "h3", "h4"):
                self.heading(node, min(int(tag[1]), 3))
            elif tag == "p":
                self.paragraph(node)
            elif tag == "blockquote":
                self.quote(node)
            elif tag == "pre":
                self.code(node)
            elif tag in ("ul", "ol"):
                self.listing(node, ordered=tag == "ol")
            elif tag == "table":
                self.table(node)
            elif tag == "hr":
                self.rule()


def build(md_text):
    html = markdown.markdown(
        preprocess(md_text), extensions=["tables", "fenced_code"])
    html = BLOCKQUOTE_FENCE.sub(r"<pre><code>\1</code></pre>", html)
    root = ET.fromstring(f"<root>{html}</root>")

    doc = Document()
    normal = doc.styles["Normal"]
    normal.font.size = Pt(10)
    normal.font.name = BODY_FONT
    normal.element.rPr.rFonts.set(qn("w:eastAsia"), BODY_FONT)
    normal.paragraph_format.space_after = Pt(4)
    normal.paragraph_format.line_spacing = 1.15

    Builder(doc).walk(root)
    return doc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    args = ap.parse_args()

    with open(args.src, encoding="utf-8") as fh:
        doc = build(fh.read())
    doc.save(args.out)
    print(f"{args.out}  {os.path.getsize(args.out)/1024:.0f} KB")


if __name__ == "__main__":
    main()
