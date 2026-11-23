"""Synthetic *scanned technical book* with ground truth.

Builds a small, realistic computer book with ReportLab: chapter openers,
numbered sections, indented justified paragraphs, code listings on a grey
panel with captions, a NOTE box, bulleted and numbered lists, a captioned
figure and table, running heads and folios. The vector PDF is then
rasterised and degraded like a cheap scan (skew, yellow paper, fading,
noise), giving exactly the input Mode B has to rebuild - plus the ground
truth to measure how well it did.
"""

from __future__ import annotations

import io
import random

import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (BaseDocTemplate, Flowable, Frame, Image, KeepTogether, ListFlowable,
                                ListItem, PageBreak, PageTemplate, Paragraph, Preformatted, Spacer, Table,
                                TableStyle)

import synth

PAGE = (7 * inch, 9.19 * inch)

SENTENCES = [
    "A scanned document is only as useful as the text that can be recovered from it.",
    "Every page passes through the same sequence of steps before it reaches the output.",
    "The first step measures how far the page is rotated and corrects the angle.",
    "Background normalization removes the yellow tint of old paper and uneven lighting.",
    "Adaptive thresholding then separates the printed characters from the paper.",
    "Because each page is independent, the work can be spread across several processes.",
    "This keeps memory usage flat even when the document has hundreds of pages.",
    "Reading order matters as much as recognition accuracy for multi-column layouts.",
    "Tables and code listings need special treatment because their spacing carries meaning.",
    "Small mistakes in the early stages tend to multiply in the later ones.",
    "The configuration object collects every option in a single place.",
    "In practice the default settings work well for most books and reports.",
    "We will look at each of these stages in detail in the following sections.",
    "The results are written to a temporary directory and assembled at the end.",
    "Careful engineering of these details is what separates a toy from a tool.",
]

LISTING_1 = '''def process(pages, workers=4):
    """Deskew and clean every page."""
    results = []
    for page in pages:
        if page.is_blank():
            continue
        angle = estimate_skew(page)
        results.append(rotate(page, -angle))
    return results'''

LISTING_2 = '''class Pipeline:
    def __init__(self, config):
        self.config = config
        self.stages = [render, deskew, enhance]

    def run(self, path):
        for stage in self.stages:
            path = stage(path, self.config)
        return path'''


def _paras(rng: random.Random, n: int, k=(3, 5)) -> list[str]:
    return [" ".join(rng.sample(SENTENCES, rng.randint(*k))) for _ in range(n)]


class ChapterOpener(Flowable):
    """'CHAPTER 1' label + big title, like a real chapter opening page."""

    def __init__(self, number: int, title: str):
        super().__init__()
        self.number, self.title = number, title

    def wrap(self, aw, ah):
        self.aw = aw
        return aw, 1.9 * inch

    def draw(self):
        c = self.canv
        c.setFont("Helvetica-Bold", 13)
        c.drawString(0, 1.55 * inch, f"CHAPTER {self.number}")
        c.setFont("Helvetica-Bold", 25)
        c.drawString(0, 1.05 * inch, self.title)
        c.setLineWidth(1.2)
        c.line(0, 0.85 * inch, self.aw, 0.85 * inch)


def build_vector_book(seed: int = 0) -> tuple[bytes, dict]:
    """Return (vector PDF bytes, ground truth)."""
    rng = random.Random(seed)
    body = ParagraphStyle("body", fontName="Times-Roman", fontSize=10.5, leading=13.5,
                          alignment=TA_JUSTIFY, firstLineIndent=14, spaceAfter=0)
    body_first = ParagraphStyle("bf", parent=body, firstLineIndent=0)
    h1 = ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=14, leading=18, spaceBefore=14, spaceAfter=6)
    cap = ParagraphStyle("cap", fontName="Helvetica-Bold", fontSize=9, leading=12, spaceBefore=8, spaceAfter=4)
    code = ParagraphStyle("code", fontName="Courier", fontSize=9, leading=11.5)
    note = ParagraphStyle("note", fontName="Times-Roman", fontSize=10, leading=13)

    gt = {"chapters": [], "sections": [], "paragraphs": [], "code": [], "captions": [],
          "bullets": [], "numbered": [], "notes": []}
    story: list = []
    state = {"chapter": "", "section": ""}

    class Mark(Flowable):  # zero-size marker that updates the running heads
        def __init__(self, key, val):
            super().__init__()
            self.key, self.val = key, val

        def wrap(self, *a):
            return 0, 0

        def draw(self):
            state[self.key] = self.val

    def chapter(n, title):
        if story:
            story.append(PageBreak())
        story.append(Mark("chapter", f"{n}  {title}"))
        story.append(ChapterOpener(n, title))
        gt["chapters"].append(title)

    def section(num, title, style=h1):
        story.append(Mark("section", f"{num} {title}"))
        story.append(Paragraph(f"{num}&nbsp;&nbsp;{title}", style))
        gt["sections"].append((num, title))

    def paragraphs(n):
        for i, t in enumerate(_paras(rng, n)):
            story.append(Paragraph(t, body_first if i == 0 else body))
            gt["paragraphs"].append(t)

    def listing(caption, text):
        gt["code"].append(text)
        gt["captions"].append(caption)
        panel = Table([[Preformatted(text, code)]], colWidths=[PAGE[0] - 1.9 * inch])
        panel.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.Color(0.86, 0.86, 0.86)),
                                   ("LEFTPADDING", (0, 0), (-1, -1), 8), ("TOPPADDING", (0, 0), (-1, -1), 6),
                                   ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
        story.append(KeepTogether([Paragraph(caption, cap), panel, Spacer(1, 8)]))

    def bullets(items, numbered=False):
        gt["numbered" if numbered else "bullets"].extend(items)
        story.append(ListFlowable([ListItem(Paragraph(t, body_first), leftIndent=18) for t in items],
                                  bulletType="1" if numbered else "bullet", start="1" if numbered else "•",
                                  bulletFontName="Times-Roman", leftIndent=18, spaceBefore=4, spaceAfter=4))

    def note_box(text):
        gt["notes"].append(text)
        box = Table([[Paragraph(f"<b>NOTE</b>&nbsp;&nbsp;{text}", note)]], colWidths=[PAGE[0] - 2.1 * inch])
        box.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.9, colors.black),
                                 ("LEFTPADDING", (0, 0), (-1, -1), 8), ("TOPPADDING", (0, 0), (-1, -1), 6),
                                 ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]))
        story.extend([Spacer(1, 6), box, Spacer(1, 6)])

    def figure(caption):
        gt["captions"].append(caption)
        yy, xx = np.mgrid[0:220, 0:420]
        img = np.full((220, 420, 3), 255, np.uint8)
        for i, x in enumerate((20, 160, 300)):  # three boxes and arrows: a diagram
            img[70:150, x:x + 100] = (150 + 30 * i, 180, 220 - 40 * i)
            if i < 2:
                img[106:114, x + 100:x + 140] = 40
        buf = io.BytesIO()
        from PIL import Image as PILImage

        PILImage.fromarray(img).save(buf, "PNG")
        buf.seek(0)
        story.append(KeepTogether([Spacer(1, 6), Image(buf, width=4.2 * inch, height=2.2 * inch),
                                   Paragraph(caption, cap)]))

    def table(caption, rows):
        gt["captions"].append(caption)
        t = Table(rows, colWidths=[1.6 * inch, 1.1 * inch, 2.0 * inch])
        t.setStyle(TableStyle([("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 9),
                               ("FONT", (0, 1), (-1, -1), "Times-Roman", 9.5),
                               ("GRID", (0, 0), (-1, -1), 0.7, colors.black),
                               ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
        story.append(KeepTogether([Paragraph(caption, cap), t, Spacer(1, 8)]))

    # ----------------------------------------------------------------- content
    chapter(1, "Getting started with pipelines")
    paragraphs(2)
    section("1.1", "Why pipelines matter")
    paragraphs(3)
    bullets(["Rendering converts each page to an image.",
             "Deskewing straightens the text lines.",
             "Enhancement makes faded text crisp again."])
    paragraphs(1)
    listing("Listing 1.1 A simple processing loop", LISTING_1)
    paragraphs(2)
    note_box("The skew angle is estimated on a downscaled copy of the page, which is much faster.")
    paragraphs(2)
    section("1.2", "Configuring the stages")
    paragraphs(2)
    table("Table 1.1 Default settings", [["Option", "Default", "Meaning"], ["dpi", "300", "Render resolution"],
                                         ["workers", "4", "Parallel processes"], ["batch_size", "8", "Pages in flight"]])
    paragraphs(3)
    chapter(2, "Building the processing engine")
    paragraphs(2)
    section("2.1", "The pipeline object")
    paragraphs(2)
    listing("Listing 2.1 The pipeline class", LISTING_2)
    paragraphs(2)
    figure("Figure 2.1 The three main stages of the engine")
    paragraphs(2)
    section("2.2", "Running the engine")
    bullets(["Create a configuration.", "Open the input document.", "Call the run method."], numbered=True)
    paragraphs(3)

    # ------------------------------------------------------------------- pages
    def on_page(c, doc):
        n = doc.page
        c.saveState()
        c.setFont("Times-Roman", 9)
        first_of_chapter = any(isinstance(f, ChapterOpener) for f in getattr(doc, "_this_page", []))
        if not first_of_chapter and n > 1:
            if n % 2 == 0:
                c.drawString(0.85 * inch, PAGE[1] - 0.55 * inch, f"{n}")
                c.drawString(1.3 * inch, PAGE[1] - 0.55 * inch, f"CHAPTER {state['chapter']}")
            else:
                c.drawRightString(PAGE[0] - 0.85 * inch, PAGE[1] - 0.55 * inch, f"{n}")
                c.drawRightString(PAGE[0] - 1.3 * inch, PAGE[1] - 0.55 * inch, state["section"])
        else:
            c.drawCentredString(PAGE[0] / 2, 0.5 * inch, str(n))
        c.restoreState()

    buf = io.BytesIO()
    doc = BaseDocTemplate(buf, pagesize=PAGE, leftMargin=0.85 * inch, rightMargin=0.85 * inch,
                          topMargin=0.85 * inch, bottomMargin=0.8 * inch)
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="f")
    doc.addPageTemplates([PageTemplate("p", [frame], onPageEnd=on_page)])

    orig_handle = doc.handle_flowable

    def track(flowables):  # remember flowables placed on the current page
        if flowables and isinstance(flowables[0], ChapterOpener):
            doc._this_page = [flowables[0]]
        return orig_handle(flowables)

    doc.handle_flowable = track
    orig_new = doc.handle_pageBegin

    def page_begin():
        doc._this_page = []
        return orig_new()

    doc.handle_pageBegin = page_begin
    doc.build(story)
    return buf.getvalue(), gt


def scanned_book(seed: int = 0, dpi: int = 200, max_angle: float = 2.5) -> tuple[bytes, dict]:
    """Degraded image-only PDF of the synthetic book + ground truth."""
    import pymupdf as fitz

    vec, gt = build_vector_book(seed)
    rng = random.Random(seed)
    imgs = []
    with fitz.open(stream=vec, filetype="pdf") as d:
        for i, page in enumerate(d):
            pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csRGB)
            img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3).copy()
            imgs.append(synth.degrade(img, angle=round(rng.uniform(-max_angle, max_angle), 1), seed=seed + i))
        gt["pages"] = d.page_count
    return synth.images_to_pdf(imgs, dpi), gt
