"""Render a ``structure.Book`` as a professionally typeset PDF.

The book is written as semantic HTML (chapters, headings, paragraphs, code
listings, callouts, figures, tables) and laid out by WeasyPrint with the
CSS paged-media stylesheet ``book.css``: trim size with mirrored margins,
running heads (chapter on the verso, section on the recto), chapter
openers on right-hand pages, a generated table of contents with page
numbers, PDF bookmarks, Knuth-style hyphenation (Pyphen) and proper
kerning/ligatures (Pango/HarfBuzz).
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..structure import Book, Node

log = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent

# name: (width, height, inner, outer, top, bottom)  -- inches
TRIMS = {
    "7x9.19in": (7.0, 9.19, 0.9, 0.72, 0.85, 0.8),     # classic technical-book trim
    "7.375x9.25in": (7.375, 9.25, 0.95, 0.75, 0.85, 0.8),
    "A4": (8.27, 11.69, 1.0, 0.85, 1.0, 0.95),
    "Letter": (8.5, 11.0, 1.0, 0.9, 0.95, 0.9),
}


def weasyprint_available() -> tuple[bool, str]:
    try:
        import weasyprint  # noqa: F401

        return True, weasyprint.__version__
    except Exception as exc:  # noqa: BLE001 - OSError when Pango is missing
        return False, str(exc).splitlines()[0][:200]


# ---------------------------------------------------------------- typography
_QUOTE_OPEN = re.compile(r'(^|[\s(\[{\u2014\u2013-])"')
_SQUOTE_OPEN = re.compile(r"(^|[\s(\[{\u2014\u2013-])'")


def smarten(text: str) -> str:
    """Typographic quotes, apostrophes and ellipses for *prose* (never code)."""
    text = _QUOTE_OPEN.sub("\\1\u201c", text)
    text = text.replace('"', "\u201d")
    text = _SQUOTE_OPEN.sub("\\1\u2018", text)
    text = text.replace("'", "\u2019")
    text = text.replace("...", "\u2026")
    text = re.sub(r"(?<=\w) -- (?=\w)", "\u2014", text)
    return text


def _slug(text: str, used: set) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40] or "s"
    s, k = base, 2
    while s in used:
        s, k = f"{base}-{k}", k + 1
    used.add(s)
    return s


@dataclass
class _Sec:
    chapter: Optional[Node]
    nodes: list
    id: str = ""


def _prepare(book: Book) -> tuple[list[_Sec], list[dict]]:
    """Split nodes into chapters, give headings ids, build the TOC."""
    used: set = set()
    sections: list[_Sec] = [_Sec(chapter=None, nodes=[])]
    toc: list[dict] = []
    for n in book.nodes:
        if n.kind == "chapter":
            sec = _Sec(chapter=n, nodes=[], id=_slug(f"ch-{n.number}-{n.text}", used))
            sections.append(sec)
            toc.append({"id": sec.id, "number": n.number, "title": n.text, "sections": []})
            continue
        if n.kind == "heading":
            n.id = _slug(f"{n.number}-{n.text}", used)
            if toc and n.level <= 3:
                toc[-1]["sections"].append({"id": n.id, "number": n.number, "title": n.text, "level": n.level})
        if n.kind in ("para", "callout"):
            n.text = smarten(n.text)
        if n.kind == "list":
            n.items = [smarten(i) for i in n.items]
        if n.kind in ("code", "figure", "table") and n.caption:
            lab = f"{n.label} {n.number}"
            n.caption_text = smarten(n.caption[len(lab):].strip() if n.caption.startswith(lab) else n.caption)
        else:
            n.caption_text = ""
        if n.kind == "figure" and n.src:
            n.src_url = Path(n.src).resolve().as_uri()
        if n.kind == "table" and n.rows:
            width = max(len(r) for r in n.rows)
            n.rows = [list(r) + [""] * (width - len(r)) for r in n.rows]
        sections[-1].nodes.append(n)
    if not sections[0].nodes:
        sections = sections[1:]
    return sections, toc


def _page_css(trim: str) -> str:
    w, h, inner, outer, top, bottom = TRIMS.get(trim, TRIMS["7x9.19in"])
    return (f"@page {{ size: {w}in {h}in; margin: {top}in {outer}in {bottom}in {inner}in; }}\n"
            f"@page :left {{ margin-left: {outer}in; margin-right: {inner}in; }}\n"
            f"@page :right {{ margin-left: {inner}in; margin-right: {outer}in; }}\n")


def render_html(book: Book, trim: str = "7x9.19in", chapters_on_right: bool = True,
                lang: str = "en", front_matter: bool = True) -> str:
    from jinja2 import Environment, FileSystemLoader, select_autoescape

    env = Environment(loader=FileSystemLoader(str(HERE)), autoescape=select_autoescape(["html", "j2"]),
                      trim_blocks=True, lstrip_blocks=True)
    sections, toc = _prepare(book)
    css = _page_css(trim) + (HERE / "book.css").read_text(encoding="utf-8")
    return env.get_template("book.html.j2").render(
        book=book, sections=sections, toc=toc, css=css, lang=lang, front_matter=front_matter,
        chapters_on_right=chapters_on_right, trim_class=re.sub(r"[^a-z0-9]", "", trim.lower()))


def render_pdf(book: Book, output_path: str, trim: str = "7x9.19in", chapters_on_right: bool = True,
               lang: str = "en", html_out: Optional[str] = None, front_matter: bool = True) -> str:
    """Typeset ``book`` to ``output_path`` (raises ImportError/OSError if
    WeasyPrint or its native Pango libraries are unavailable)."""
    import weasyprint
    from weasyprint.text.fonts import FontConfiguration

    # WeasyPrint is chatty about optional native subsetting libraries.
    logging.getLogger("weasyprint").setLevel(logging.ERROR)
    logging.getLogger("fontTools").setLevel(logging.ERROR)

    doc_html = render_html(book, trim=trim, chapters_on_right=chapters_on_right, lang=lang,
                           front_matter=front_matter)
    if html_out:
        Path(html_out).write_text(doc_html, encoding="utf-8")
    fonts = FontConfiguration()
    weasyprint.HTML(string=doc_html, base_url=str(HERE) + os.sep).write_pdf(
        output_path, font_config=fonts, optimize_images=False)
    return output_path
