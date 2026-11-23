"""Output PDF builders.

Mode A — ``VisualPDFBuilder`` (PyMuPDF)
    Keeps the original look: each cleaned, deskewed page image is placed on a
    page and an *invisible* OCR text layer (render mode 3) is laid exactly over
    the words, so the PDF is searchable / selectable / copyable. Pages are
    appended incrementally and the document is periodically flushed to disk
    (incremental save + reopen) so memory stays flat for 1000-page books.

Mode B — ``ReflowPDFBuilder`` (ReportLab Platypus)
    Rebuilds the document from the layout model: headings, justified
    paragraphs, bullet lists, real tables and cropped figures on standard
    A4/Letter pages with consistent typography. Running headers/footers and
    page numbers from the scan are detected across pages and dropped.

Both builders share the same tiny interface:  ``add_page(result)``, ``close()``.
"""

from __future__ import annotations

import html
import logging
import os
import re
from collections import Counter
from typing import Optional

import pymupdf as fitz

from .config import PipelineConfig
from .models import Block, PageLayout, PageResult

log = logging.getLogger(__name__)

PAGE_SIZES_PT = {"A4": (595.276, 841.890), "Letter": (612.0, 792.0)}

# Unicode TrueType fonts, best first. DejaVu is preferred everywhere (on
# macOS: `brew install --cask font-dejavu`, which installs to ~/Library/Fonts);
# otherwise fall back to fonts that ship with the OS.
_FONT_DIRS_DEJAVU = [
    "/usr/share/fonts/truetype/dejavu",         # Debian / Ubuntu
    "/usr/share/fonts/dejavu",                  # Fedora / Arch
    "/usr/share/fonts/dejavu-sans-fonts",       # Fedora (newer)
    "~/Library/Fonts",                          # macOS, per-user (Homebrew cask)
    "/Library/Fonts",                           # macOS, system-wide
    "/opt/homebrew/share/fonts",                # macOS, Homebrew (Apple Silicon)
    "/usr/local/share/fonts",                   # macOS (Intel Homebrew) / Linux
]
_FONT_FALLBACKS = {
    False: [
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",  # macOS 10.15+
        "/Library/Fonts/Arial Unicode.ttf",                      # older macOS
        "C:/Windows/Fonts/arial.ttf",
    ],
    True: [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",     # macOS 10.15+
        "/Library/Fonts/Arial Bold.ttf",                         # older macOS
        "C:/Windows/Fonts/arialbd.ttf",
    ],
}


def find_unicode_font(bold: bool = False) -> Optional[str]:
    """Path of a Unicode TTF for the text layer / reflow, or None (Helvetica)."""
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    candidates = [os.path.join(os.path.expanduser(d), name) for d in _FONT_DIRS_DEJAVU]
    for p in candidates + _FONT_FALLBACKS[bold]:
        if os.path.exists(p):
            return p
    return None


def make_builder(cfg: PipelineConfig, output_path: str, work_dir: str):
    if cfg.mode == "B":
        return BookBuilder(output_path, cfg, work_dir)
    return VisualPDFBuilder(output_path, cfg, work_dir)


# =============================================================================
# Mode A: visual deskew + invisible OCR layer
# =============================================================================

class VisualPDFBuilder:
    FLUSH_EVERY = 25  # pages between incremental saves

    def __init__(self, output_path: str, cfg: PipelineConfig, work_dir: str):
        self.output_path = output_path
        self.cfg = cfg
        self.partial = os.path.join(work_dir, "_partial_A.pdf")
        self.doc = fitz.open()
        self.pending = 0
        self.flushed = False
        self.pages = 0
        self._helv = fitz.Font("helv")
        uni = find_unicode_font()
        self._uni = fitz.Font(fontfile=uni) if uni else None

    # -- page geometry ----------------------------------------------------
    def _page_rect(self, w_px: int, h_px: int, dpi: int) -> tuple[fitz.Rect, fitz.Rect]:
        """Return (page rect, image rect) in points."""
        w_pt, h_pt = w_px * 72.0 / dpi, h_px * 72.0 / dpi
        if self.cfg.page_size == "original":
            r = fitz.Rect(0, 0, w_pt, h_pt)
            return r, r
        pw, ph = PAGE_SIZES_PT[self.cfg.page_size]
        if w_pt > h_pt:  # landscape content -> landscape page
            pw, ph = ph, pw
        s = min(pw / w_pt, ph / h_pt)
        iw, ih = w_pt * s, h_pt * s
        x0, y0 = (pw - iw) / 2, (ph - ih) / 2
        return fitz.Rect(0, 0, pw, ph), fitz.Rect(x0, y0, x0 + iw, y0 + ih)

    # -- public API -------------------------------------------------------
    def add_page(self, res: PageResult) -> None:
        if res.status == "blank":
            if self.cfg.blank_policy == "drop":
                return
            w, h = (res.width_px or 2480), (res.height_px or 3508)
            page_rect, _ = self._page_rect(w, h, res.dpi)
            self.doc.new_page(width=page_rect.width, height=page_rect.height)
        elif not res.image_path:
            self._placeholder(res)
        else:
            page_rect, img_rect = self._page_rect(res.width_px, res.height_px, res.dpi)
            page = self.doc.new_page(width=page_rect.width, height=page_rect.height)
            page.insert_image(img_rect, filename=res.image_path)
            layout = res.load_layout()
            if layout is not None:
                self._text_layer(page, layout, img_rect)
        self.pages += 1
        self.pending += 1
        if self.pending >= self.FLUSH_EVERY:
            self._flush()

    def _placeholder(self, res: PageResult) -> None:
        w, h = PAGE_SIZES_PT["A4"] if self.cfg.page_size == "original" else PAGE_SIZES_PT[self.cfg.page_size]
        page = self.doc.new_page(width=w, height=h)
        page.insert_textbox(
            fitz.Rect(72, h / 2 - 60, w - 72, h / 2 + 60),
            f"Page {res.index + 1} could not be processed.\n{res.message[:200]}",
            fontsize=11, fontname="helv", color=(0.5, 0.1, 0.1), align=1,
        )

    def _text_layer(self, page, layout: PageLayout, img_rect: fitz.Rect) -> None:
        """Invisible, word-accurate text layer.

        Each word is written with a font size chosen so its advance width
        matches the OCR box width (selection highlights line up with the
        visible glyphs), clamped against the box height.
        """
        sx = img_rect.width / max(layout.width, 1)
        sy = img_rect.height / max(layout.height, 1)
        tw = fitz.TextWriter(page.rect)
        for w in layout.all_words():
            text = w.text
            x0, y0, x1, y1 = w.bbox
            bw, bh = (x1 - x0) * sx, (y1 - y0) * sy
            if bw <= 0 or bh <= 0:
                continue
            font = self._helv
            try:
                text.encode("latin-1")
            except UnicodeEncodeError:
                font = self._uni or self._helv
            fs_h = bh * 0.95
            adv = font.text_length(text, fontsize=fs_h)
            fs = fs_h * (bw / adv) if adv > 0 else fs_h
            fs = max(min(fs, fs_h * 1.6), fs_h * 0.4, 1.0)
            origin = fitz.Point(img_rect.x0 + x0 * sx, img_rect.y0 + y1 * sy - 0.18 * bh)
            try:
                tw.append(origin, text + " ", font=font, fontsize=fs)
            except Exception:  # noqa: BLE001 - unsupported glyphs etc.
                continue
        tw.write_text(page, render_mode=3)  # 3 = invisible

    def _flush(self) -> None:
        """Write pages to disk and reopen so MuPDF can release memory."""
        if not self.flushed:
            self.doc.save(self.partial, garbage=1, deflate=True)
            self.flushed = True
        else:
            self.doc.saveIncr()
        self.doc.close()
        self.doc = fitz.open(self.partial)
        self.pending = 0

    def close(self) -> str:
        if self.doc.page_count == 0:
            self.doc.new_page()  # a PDF must have at least one page
        if self.flushed:
            if self.pending:
                self.doc.saveIncr()
            self.doc.close()
            self.doc = fitz.open(self.partial)
        try:
            self.doc.subset_fonts()
        except Exception:  # noqa: BLE001 - optional optimisation
            pass
        self.doc.set_metadata({"producer": "pdf-align (PyMuPDF)", "creator": "pdf-align Mode A"})
        self.doc.save(self.output_path, garbage=3, deflate=True, clean=False)
        self.doc.close()
        if os.path.exists(self.partial):
            os.remove(self.partial)
        return self.output_path


# =============================================================================
# Mode B: full reflow
# =============================================================================

_PAGE_NUM_RE = re.compile(r"^\s*(page\s*)?[-–—]?\s*[\divxlcIVXLC]{1,6}\s*[-–—]?\s*(of\s*\d+)?\s*$", re.I)
_DIGITS_RE = re.compile(r"\d+")


def _norm_running(text: str) -> str:
    return _DIGITS_RE.sub("#", text.lower()).strip()


def join_lines(lines: list[str]) -> str:
    """Join OCR lines into flowing text with de-hyphenation.

    'infor-' + 'mation' -> 'information'; but 'well-' + 'Known' stays hyphenated.
    """
    out = ""
    for ln in (l.strip() for l in lines):
        if not ln:
            continue
        if not out:
            out = ln
        elif out.endswith("-") and len(out) > 1 and out[-2].isalpha() and ln[:1].islower():
            out = out[:-1] + ln
        else:
            out += " " + ln
    return re.sub(r"\s{2,}", " ", out)


class ReflowPDFBuilder:
    """Collect lightweight flowables page by page and typeset once at the end.

    Only paths and short strings are retained; figure bitmaps are read from
    disk lazily by ReportLab while it lays out the document.
    """

    def __init__(self, output_path: str, cfg: PipelineConfig, work_dir: str, page_breaks: bool = False):
        self.output_path = output_path
        self.cfg = cfg
        self.work_dir = work_dir
        self.page_breaks = page_breaks
        self.results: list[PageResult] = []

    def add_page(self, res: PageResult) -> None:
        # Mode B needs a cross-page pass (running headers, paragraphs split
        # over page breaks), so we only keep the on-disk handles here.
        self.results.append(res)

    # -- typography -------------------------------------------------------
    def _styles(self):
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont

        regular, bold = "Helvetica", "Helvetica-Bold"
        reg_path, bold_path = find_unicode_font(), find_unicode_font(bold=True)
        if reg_path:
            try:
                pdfmetrics.registerFont(TTFont("BodyFont", reg_path))
                regular = "BodyFont"
                if bold_path and os.path.exists(bold_path):
                    pdfmetrics.registerFont(TTFont("BodyFont-Bold", bold_path))
                    bold = "BodyFont-Bold"
                else:
                    bold = regular
            except Exception as exc:  # noqa: BLE001
                log.warning("could not register %s: %s", reg_path, exc)

        base = getSampleStyleSheet()
        ink = colors.HexColor("#1d1d1f")
        st = {
            "body": ParagraphStyle("body", parent=base["BodyText"], fontName=regular, fontSize=10.5,
                                   leading=15.2, alignment=TA_JUSTIFY, spaceAfter=7, textColor=ink),
            "h1": ParagraphStyle("h1", parent=base["Heading1"], fontName=bold, fontSize=19, leading=24,
                                 spaceBefore=14, spaceAfter=10, textColor=colors.HexColor("#0b2545")),
            "h2": ParagraphStyle("h2", parent=base["Heading2"], fontName=bold, fontSize=14.5, leading=19,
                                 spaceBefore=12, spaceAfter=6, textColor=colors.HexColor("#13315c")),
            "h3": ParagraphStyle("h3", parent=base["Heading3"], fontName=bold, fontSize=11.5, leading=15,
                                 spaceBefore=10, spaceAfter=4, textColor=colors.HexColor("#13315c")),
            "list": ParagraphStyle("list", fontName=regular, fontSize=10.5, leading=15, leftIndent=18,
                                   bulletIndent=6, spaceAfter=3, alignment=TA_LEFT, textColor=ink),
            "cell": ParagraphStyle("cell", fontName=regular, fontSize=9, leading=11.5, textColor=ink),
            "cellh": ParagraphStyle("cellh", fontName=bold, fontSize=9, leading=11.5, textColor=ink),
            "note": ParagraphStyle("note", fontName=regular, fontSize=9, leading=12,
                                   textColor=colors.HexColor("#8a1c1c"), alignment=TA_LEFT),
            "_regular": regular,
        }
        return st

    # -- running header/footer detection ----------------------------------
    def _running_texts(self, layouts: list[Optional[PageLayout]]) -> set[str]:
        counts: Counter = Counter()
        for lay in layouts:
            if lay is None:
                continue
            for b in lay.blocks:
                if b.type in ("header", "footer"):
                    counts[_norm_running(b.text)] += 1
        n = sum(1 for l in layouts if l is not None)
        return {t for t, c in counts.items() if c >= max(2, int(0.3 * n))}

    def _is_running(self, b: Block, running: set[str]) -> bool:
        t = b.text.strip()
        return not t or bool(_PAGE_NUM_RE.match(t)) or _norm_running(t) in running

    # -- block -> flowables -----------------------------------------------
    def _table(self, b: Block, st, frame_w: float):
        from reportlab.lib import colors
        from reportlab.platypus import Paragraph, Table, TableStyle

        cells = b.cells or []
        ncols = max(len(r) for r in cells)
        data = []
        for r_i, row in enumerate(cells):
            row = list(row) + [""] * (ncols - len(row))
            style = st["cellh"] if r_i == 0 else st["cell"]
            data.append([Paragraph(html.escape(c), style) for c in row])
        # Column widths proportional to content length, with a floor.
        lens = [max(4, max(len(r[j]) if j < len(r) else 0 for r in cells)) for j in range(ncols)]
        total = float(sum(lens))
        widths = [max(frame_w * 0.08, frame_w * l / total) for l in lens]
        k = frame_w / sum(widths)
        widths = [w * k for w in widths]
        t = Table(data, colWidths=widths, repeatRows=1, hAlign="CENTER")
        t.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#9aa5b1")),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef5")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7f9fb")]),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        return t

    def _figure(self, path: str, frame_w: float, frame_h: float):
        from PIL import Image as PILImage
        from reportlab.platypus import Image

        with PILImage.open(path) as im:
            w, h = im.size
        s = min(frame_w / w, (frame_h * 0.6) / h, 1.0 * 72 / 150)  # never upscale beyond 150 dpi
        img = Image(path, width=w * s, height=h * s, lazy=2)
        img.hAlign = "CENTER"
        return img

    def _block_flowables(self, b: Block, st, frame_w, frame_h, pending_para: list):
        from reportlab.platypus import Paragraph, Spacer

        out = []
        if b.type == "heading":
            style = st["h%d" % min(max(b.level, 1), 3)]
            out.append(Paragraph(html.escape(join_lines([l.text for l in b.lines])), style))
        elif b.type == "list":
            items: list[list[str]] = []
            from .layout_engine import BULLET_RE

            for l in b.lines:
                if BULLET_RE.match(l.text) or not items:
                    items.append([BULLET_RE.sub("", l.text, count=1)])
                else:
                    items[-1].append(l.text)
            for it in items:
                out.append(Paragraph(html.escape(join_lines(it)), st["list"], bulletText="\u2022"))
        elif b.type == "table" and b.cells:
            out.append(Spacer(1, 4))
            out.append(self._table(b, st, frame_w))
            out.append(Spacer(1, 8))
        elif b.type == "figure" and b.image_path and os.path.exists(b.image_path):
            out.append(Spacer(1, 4))
            out.append(self._figure(b.image_path, frame_w, frame_h))
            out.append(Spacer(1, 8))
        elif b.lines:  # paragraph (or kept header/footer)
            pending_para.append(join_lines([l.text for l in b.lines]))
            return out, True
        return out, False

    def close(self) -> str:
        from reportlab.lib.units import cm
        from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate

        size = PAGE_SIZES_PT.get(self.cfg.page_size) or PAGE_SIZES_PT.get(self.cfg.trim) or PAGE_SIZES_PT["A4"]
        margin = 2.2 * cm
        st = self._styles()
        frame_w, frame_h = size[0] - 2 * margin, size[1] - 2 * margin

        layouts = [r.load_layout() if r.status == "ok" else None for r in self.results]
        running = self._running_texts(layouts)

        story: list = []
        para_buf: list[str] = []  # paragraph text that may continue on next page

        def flush_para():
            if para_buf:
                story.append(Paragraph(html.escape(" ".join(para_buf)), st["body"]))
                para_buf.clear()

        for res, lay in zip(self.results, layouts):
            if res.status == "blank":
                continue
            if lay is None:
                flush_para()
                if res.image_path and os.path.exists(res.image_path):
                    story.append(Paragraph(f"[Page {res.index + 1} could not be re-typeset; "
                                           f"original image follows]", st["note"]))
                    story.append(self._figure(res.image_path, frame_w, frame_h * 1.6))
                else:
                    story.append(Paragraph(f"[Page {res.index + 1} could not be processed: "
                                           f"{html.escape(res.message[:150])}]", st["note"]))
                continue
            for b in lay.blocks:
                if b.type in ("header", "footer") and self._is_running(b, running):
                    continue
                is_text = b.type in ("paragraph", "header", "footer") and bool(b.lines)
                if is_text:
                    text = join_lines([l.text for l in b.lines])
                    # Paragraph continues from previous page/column if the
                    # previous chunk didn't end a sentence and this starts
                    # lower-case.
                    if para_buf and not re.search(r"[.!?:;\"')\]]$", para_buf[-1]) and text[:1].islower():
                        if para_buf[-1].endswith("-") and para_buf[-1][-2:-1].isalpha():
                            para_buf[-1] = para_buf[-1][:-1] + text
                        else:
                            para_buf.append(text)
                        continue
                    flush_para()
                    para_buf.append(text)
                    continue
                flush_para()
                flows, _ = self._block_flowables(b, st, frame_w, frame_h, para_buf)
                story.extend(flows)
            if self.page_breaks:
                flush_para()
                story.append(PageBreak())
        flush_para()
        if not story:
            story.append(Paragraph("(No text could be recognised in this document.)", st["note"]))

        regular = st["_regular"]

        def on_page(canvas, doc):
            canvas.saveState()
            canvas.setFont(regular, 8.5)
            canvas.setFillGray(0.45)
            canvas.drawCentredString(size[0] / 2, margin * 0.5, str(doc.page))
            canvas.restoreState()

        doc = SimpleDocTemplate(self.output_path, pagesize=size, leftMargin=margin, rightMargin=margin,
                                topMargin=margin, bottomMargin=margin,
                                title="Re-typeset document", creator="pdf-align Mode B")
        doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
        return self.output_path


# =============================================================================
# Mode B (default): professional book typesetting
# =============================================================================

class BookBuilder:
    """Whole-book structure reconstruction + CSS paged-media typesetting.

    ``close()`` runs ``structure.build_book`` over every page layout and
    renders the result with WeasyPrint (``typeset.render_pdf``). If WeasyPrint
    or its native Pango libraries are unavailable, it falls back to the
    simpler ReportLab reflow so the user still gets a document.
    """

    def __init__(self, output_path: str, cfg: PipelineConfig, work_dir: str):
        self.output_path = output_path
        self.cfg = cfg
        self.work_dir = work_dir
        self.results: list[PageResult] = []
        self.engine = "weasyprint"
        self.warning = ""

    def add_page(self, res: PageResult) -> None:
        self.results.append(res)

    def close(self) -> str:
        from .structure import build_book

        book = build_book(self.results, title=self.cfg.title, author=self.cfg.author, spellfix=self.cfg.spellfix)
        with open(os.path.join(self.work_dir, "book.json"), "w", encoding="utf-8") as fh:
            fh.write(book.to_json())
        try:
            from .typeset import render_pdf

            render_pdf(book, self.output_path, trim=self.cfg.trim, chapters_on_right=self.cfg.chapters_on_right,
                       lang=(self.cfg.ocr_lang or "eng")[:2] if self.cfg.ocr_lang.startswith("eng") else "en",
                       html_out=self.cfg.html_out or None, front_matter=self.cfg.front_matter)
        except (ImportError, OSError) as exc:
            self.engine = "reportlab"
            self.warning = (f"WeasyPrint unavailable ({str(exc).splitlines()[0][:120]}); "
                            "used the basic ReportLab layout. Install Pango for book typesetting (see README).")
            log.warning(self.warning)
            fallback = ReflowPDFBuilder(self.output_path, self.cfg, self.work_dir)
            fallback.results = self.results
            fallback.close()
        return self.output_path
