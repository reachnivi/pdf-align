"""Whole-book structure reconstruction (Mode B).

Per-page OCR layouts are noisy and local: Tesseract splits paragraphs at
random, a heading is only "big" relative to *its* page, and running heads,
folios and captions look like ordinary text. This module looks at the whole
book at once and produces a clean semantic model:

    Book
     ├─ Node(kind="chapter",  number="1", text="Getting started")
     ├─ Node(kind="heading",  level=2, number="1.1", text="Why pipelines matter")
     ├─ Node(kind="para",     text="…")                    (merged across blocks/pages)
     ├─ Node(kind="list",     items=[…], ordered=False)
     ├─ Node(kind="code",     text="def f():\\n    …", caption="Listing 1.1 …")
     ├─ Node(kind="callout",  label="NOTE", text="…")
     ├─ Node(kind="figure",   src="…/page_00004_fig0.jpg", caption="Figure 2.1 …")
     └─ Node(kind="table",    rows=[[…]], caption="Table 1.1 …")

Everything here is cheap (no pixels): it runs in the parent process on the
small per-page layout JSON files after all workers are done.
"""

from __future__ import annotations

import json
import logging
import re
import statistics
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Optional

from .models import Block, Line, PageLayout, PageResult
from .textfix import fix_line, join_lines, normalize

log = logging.getLogger(__name__)

# ----------------------------------------------------------------- patterns
CHAPTER_RE = re.compile(r"^\s*(?:CHAPTER|Chapter)\s+([0-9IVXLC|l!]{1,4})\b\.?\s*(.*)$")
SECTION_RE = re.compile(r"^\s*(\d{1,2}(?:\s?\.\s?\d{1,2}){1,3})\.?\s+([A-Z\"'“(].{1,100})$")
CAPTION_RE = re.compile(
    r"^\s*(Listing|Example|Figure|Fig\.|Table)\s+([0-9IlO|]{1,3}\s?[.\-–]\s?[0-9IlO|]{1,3})\.?\s*(.*)$")
CALLOUT_RE = re.compile(r"^\s*(NOTE|TIP|WARNING|CAUTION|IMPORTANT|DEFINITION|Note|Tip|Warning|Caution)\b[:.]?\s*(.*)$",
                        re.S)
BULLET_RE = re.compile(r"^\s*([•·●▪◦■□➢►‣\-–—*»]|[eo«©@](?=\s+[A-Z0-9\"“(]))\s+(.*)$")
NUMBERED_RE = re.compile(r"^\s*\(?(\d{1,2}|[a-h])[.)]?\s+([A-Z\"“(].*)$")
FOLIO_RE = re.compile(r"^\s*[\[(]?\s*(\d{1,4}|[ivxlc]{1,6})\s*[\])]?\s*$", re.I)


@dataclass
class Node:
    kind: str
    text: str = ""
    level: int = 0
    number: str = ""
    label: str = ""
    caption: str = ""
    src: Optional[str] = None
    items: list = field(default_factory=list)
    rows: list = field(default_factory=list)
    ordered: bool = False
    page: int = 0


@dataclass
class Book:
    title: str = ""
    author: str = ""
    nodes: list[Node] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=1, ensure_ascii=False)

    def chapters(self) -> list[Node]:
        return [n for n in self.nodes if n.kind == "chapter"]


# ----------------------------------------------------------------- helpers
@dataclass
class _Ln:
    """A text line in book context."""
    page: int
    text: str
    raw: Line
    block: Block
    bid: int
    x0: float
    y0: float
    x1: float
    y1: float
    size: float
    ink: float
    page_h: float

    @property
    def gap_ref(self) -> float:
        return self.y1


def _line_text(line: Line, spellfix: bool) -> str:
    return fix_line([(w.text, w.conf) for w in line.words], enabled=spellfix)


def _norm_digits(t: str) -> str:
    return re.sub(r"[0-9|!lI]+", "#", t.lower()).strip()


def _roman_or_int(tok: str, fallback: int) -> str:
    tok = tok.replace("|", "1").replace("!", "1").replace("l", "1")
    if tok.isdigit():
        return tok
    roman = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}
    if all(c in roman for c in tok.upper()):
        total, prev = 0, 0
        for c in reversed(tok.upper()):
            v = roman[c]
            total = total - v if v < prev else total + v
            prev = max(prev, v)
        return str(total)
    return str(fallback)


def _clean_number(num: str) -> str:
    return re.sub(r"\s", "", num).replace("O", "0").replace("l", "1").replace("I", "1").replace("|", "1")


# =============================================================================
class StructureBuilder:
    def __init__(self, spellfix: bool = True):
        self.spellfix = spellfix

    # ------------------------------------------------------------- stats
    def _stats(self, pages: list[tuple[int, PageLayout]]) -> None:
        sizes, inks = [], []
        for _, lay in pages:
            for b in lay.blocks:
                if b.type in ("paragraph", "list") and not b.frame:
                    for l in b.lines:
                        if len(l.text) >= 40:
                            sizes.append(l.height)
                            inks.append(l.ink)
        self.body = float(statistics.median(sizes)) if sizes else 40.0
        self.body_ink = float(statistics.median(inks)) if inks else 0.2
        log.debug("body line height %.1f px, ink %.3f", self.body, self.body_ink)

    def _columns(self, lines: list[_Ln]) -> tuple[float, float]:
        """Left and right edge of the text block on one page (justified text
        gives a sharp right edge; first-line indents sit right of the left)."""
        long = [l for l in lines if l.size < 1.3 * self.body and l.x1 - l.x0 > 0]
        if not long:
            return 0.0, 1e9
        widest = max(l.x1 - l.x0 for l in long)
        full = [l for l in long if l.x1 - l.x0 > 0.7 * widest] or long
        left = statistics.median(sorted(l.x0 for l in full)[: max(1, len(full) // 2)])
        right = statistics.median(sorted((l.x1 for l in full), reverse=True)[: max(1, len(full) // 2)])
        return float(left), float(right)

    # ------------------------------------------------------------- main
    def build(self, results: list[PageResult], title: str = "", author: str = "") -> Book:
        pages = [(r.index, lay) for r in results if r.status == "ok" for lay in [r.load_layout()] if lay]
        self._stats(pages)
        book = Book(title=title, author=author)
        repeated = self._repeated_marginalia(pages)

        items: list = []  # sequence of _Ln (text) or Node (atomic)
        for pidx, lay in pages:
            items.extend(self._page_items(pidx, lay, repeated))

        nodes = self._classify(items)
        nodes = self._attach_captions(nodes)
        nodes = self._merge_code(nodes)
        if not book.title:
            book.title = next((n.text for n in nodes if n.kind == "chapter"), "")
        book.nodes = nodes
        return book

    # ------------------------------------------------------------- marginalia
    def _repeated_marginalia(self, pages) -> set[str]:
        c: Counter = Counter()
        for _, lay in pages:
            for b in lay.blocks:
                if b.lines and (b.bbox[3] < 0.1 * lay.height or b.bbox[1] > 0.9 * lay.height):
                    c[_norm_digits(b.text)] += 1
        n = len(pages)
        return {t for t, k in c.items() if k >= max(2, int(0.25 * n)) and len(t) > 2}

    def _is_marginal(self, ln: _Ln, next_y0: Optional[float], prev_y1: Optional[float],
                     repeated: set[str]) -> bool:
        """Running head / folio? Position + content + isolation test."""
        h = ln.page_h
        t = ln.text.strip()
        top, bottom = ln.y1 < 0.085 * h, ln.y0 > 0.905 * h
        if not (top or bottom):
            return False
        if FOLIO_RE.match(t):
            return True
        if _norm_digits(t) in repeated:
            return True
        isolated = ((next_y0 is not None and next_y0 - ln.y1 > 1.2 * self.body) if top else
                    (prev_y1 is not None and ln.y0 - prev_y1 > 1.2 * self.body))
        has_num = bool(re.match(r"^\s*\d{1,4}\s", t) or re.search(r"\s\d{1,4}\s*$", t))
        small = ln.size <= 1.15 * self.body
        if isolated and small and (has_num or "CHAPTER" in t.upper() or SECTION_RE.match(t)):
            return True
        return False

    def _page_items(self, pidx: int, lay: PageLayout, repeated: set[str]) -> list:
        out: list = []
        lines: list[_Ln] = []
        seq: list = []
        for bid, b in enumerate(lay.blocks):
            if b.type == "code" and (b.code or b.lines):
                seq.append(Node(kind="code", text=b.code or "\n".join(l.text for l in b.lines), page=pidx))
            elif b.type == "table" and b.cells:
                seq.append(Node(kind="table", rows=[[normalize(c) for c in r] for r in b.cells], page=pidx))
            elif b.type == "figure":
                if b.image_path:
                    seq.append(Node(kind="figure", src=b.image_path, page=pidx))
            else:
                for l in b.lines:
                    if not l.words:
                        continue
                    ln = _Ln(page=pidx, text=_line_text(l, self.spellfix), raw=l, block=b, bid=bid,
                             x0=l.bbox[0], y0=l.bbox[1], x1=l.bbox[2], y1=l.bbox[3], size=l.height,
                             ink=l.ink, page_h=lay.height)
                    lines.append(ln)
                    seq.append(ln)
        # marginalia: decide with vertical neighbours on the page
        drop = set()
        for ln in lines:
            # nearest line strictly below / above (a folio on the same baseline
            # as a running head must not count as its neighbour)
            below = [o.y0 for o in lines if o.y0 > ln.y1 - 2]
            above = [o.y1 for o in lines if o.y1 < ln.y0 + 2]
            if self._is_marginal(ln, min(below) if below else None, max(above) if above else None, repeated):
                drop.add(id(ln))
        # stray OCR crumbs (single symbols)
        for ln in lines:
            if len(re.sub(r"[^A-Za-z0-9]", "", ln.text)) == 0:
                drop.add(id(ln))
        self.page_cols = getattr(self, "page_cols", {})
        self.page_cols[pidx] = self._columns([l for l in lines if id(l) not in drop and not l.block.frame])
        for it in seq:
            if isinstance(it, _Ln) and id(it) in drop:
                continue
            out.append(it)
        return out

    # ------------------------------------------------------------- classify
    def _classify(self, items: list) -> list[Node]:
        nodes: list[Node] = []
        para: list[_Ln] = []
        lst: Optional[Node] = None
        lst_x = 0.0
        chapter_no = 0
        expect_chapter_title = False
        i = 0

        def flush_para():
            nonlocal para
            if para:
                text = join_lines([l.text for l in para])
                if text.strip():
                    nodes.append(Node(kind="para", text=text, page=para[0].page))
                para = []

        def flush_list():
            nonlocal lst
            if lst is not None:
                lst.items = [join_lines(it) for it in lst.items]
                nodes.append(lst)
                lst = None

        while i < len(items):
            it = items[i]
            if isinstance(it, Node):
                flush_para()
                flush_list()
                nodes.append(it)
                i += 1
                continue
            ln: _Ln = it
            t = ln.text.strip()
            prev = items[i - 1] if i > 0 and isinstance(items[i - 1], _Ln) else None
            left, right = self.page_cols.get(ln.page, (0.0, 1e9))
            big = ln.size >= 1.6 * self.body
            bold = ln.ink >= 1.25 * self.body_ink
            heading_like = len(t) < 110 and not re.search(r"[.,;:]$", t)

            # -- chapter label / title ------------------------------------
            m = CHAPTER_RE.match(t)
            if m and (ln.size >= 1.05 * self.body or bold) and ln.y0 < 0.45 * ln.page_h:
                flush_para()
                flush_list()
                chapter_no += 1
                num = _roman_or_int(m.group(1), chapter_no)
                chapter_no = int(num) if num.isdigit() else chapter_no
                node = Node(kind="chapter", number=num, text=m.group(2).strip(), page=ln.page)
                nodes.append(node)
                expect_chapter_title = not node.text
                i += 1
                continue
            if big and heading_like and ln.y0 < 0.5 * ln.page_h:
                flush_para()
                flush_list()
                if expect_chapter_title and nodes and nodes[-1].kind == "chapter":
                    nodes[-1].text = (nodes[-1].text + " " + t).strip()
                    # multi-line titles: absorb following big lines
                    while i + 1 < len(items) and isinstance(items[i + 1], _Ln) and \
                            items[i + 1].size >= 1.6 * self.body and items[i + 1].page == ln.page:
                        i += 1
                        nodes[-1].text += " " + items[i].text.strip()
                    expect_chapter_title = False
                elif nodes and nodes[-1].kind == "chapter" and nodes[-1].page == ln.page and \
                        not any(n.kind != "chapter" for n in nodes[-1:]):
                    nodes[-1].text = (nodes[-1].text + " " + t).strip()
                else:
                    chapter_no += 1
                    nodes.append(Node(kind="chapter", number=str(chapter_no), text=t, page=ln.page))
                i += 1
                continue
            expect_chapter_title = False

            # -- captions ---------------------------------------------------
            m = CAPTION_RE.match(t)
            if m and len(t) < 160:
                flush_para()
                flush_list()
                kind = m.group(1).rstrip(".")
                kind = "Figure" if kind == "Fig" else kind
                nodes.append(Node(kind="caption", label=kind, number=_clean_number(m.group(2)),
                                  text=m.group(3).strip(), page=ln.page))
                i += 1
                continue

            # -- callouts (NOTE / TIP / boxed sidebars) ---------------------
            if ln.block.frame == "boxed" or (CALLOUT_RE.match(t) and (prev is None or prev.block is not ln.block)):
                flush_para()
                flush_list()
                blk = ln.block
                group = [ln]
                while i + 1 < len(items) and isinstance(items[i + 1], _Ln) and items[i + 1].block is blk:
                    i += 1
                    group.append(items[i])
                text = join_lines([g.text for g in group])
                mm = CALLOUT_RE.match(text)
                label, body = (mm.group(1).upper(), mm.group(2)) if mm else ("", text)
                if not mm and len(group) > 1 and len(group[0].text) < 60 and group[0].ink >= 1.2 * self.body_ink:
                    label, body = group[0].text.strip(), join_lines([g.text for g in group[1:]])
                nodes.append(Node(kind="callout", label=label, text=body.strip(), page=ln.page))
                i += 1
                continue

            # -- section headings -------------------------------------------
            m = SECTION_RE.match(t)
            if m and heading_like and (bold or ln.size >= 1.06 * self.body) and len(t) < 100:
                flush_para()
                flush_list()
                num = _clean_number(m.group(1))
                nodes.append(Node(kind="heading", level=min(4, num.count(".") + 1), number=num,
                                  text=m.group(2).strip(), page=ln.page))
                i += 1
                continue
            alone = (prev is None or prev.block is not ln.block) and (
                i + 1 >= len(items) or not isinstance(items[i + 1], _Ln) or items[i + 1].block is not ln.block)
            if heading_like and alone and len(t) < 80 and t[:1].isupper() and \
                    (ln.size >= 1.15 * self.body or (bold and ln.size >= 0.95 * self.body)):
                flush_para()
                flush_list()
                level = 2 if ln.size >= 1.3 * self.body else 3
                nodes.append(Node(kind="heading", level=level, text=t, page=ln.page))
                i += 1
                continue

            # -- lists --------------------------------------------------------
            mb = BULLET_RE.match(t)
            mn = NUMBERED_RE.match(t)
            numbered_ok = False
            if mn and not SECTION_RE.match(t):
                expected = 1 if lst is None or not lst.ordered else len(lst.items) + 1
                tok = mn.group(1)
                numbered_ok = (tok.isdigit() and int(tok) == expected) or (tok.isalpha() and
                                                                           ord(tok) - 96 == expected)
                if numbered_ok and lst is None:
                    # a lone "1 Something" is only a list if item 2 follows
                    numbered_ok = any(isinstance(x, _Ln) and NUMBERED_RE.match(x.text.strip())
                                      and NUMBERED_RE.match(x.text.strip()).group(1) in ("2", "b")
                                      for x in items[i + 1:i + 12])
            if mb or numbered_ok:
                flush_para()
                ordered = bool(numbered_ok and not mb)
                if lst is None or lst.ordered != ordered:
                    flush_list()
                    lst = Node(kind="list", ordered=ordered, page=ln.page)
                lst.items.append([(mb or mn).group(2)])
                lst_x = ln.x0
                i += 1
                continue
            if lst is not None:
                # continuation line: hanging indent under the item text
                if prev is not None and ln.x0 > lst_x + 0.4 * self.body and ln.page == prev.page \
                        and ln.y0 - prev.y1 < 1.0 * self.body:
                    lst.items[-1].append(t)
                    i += 1
                    continue
                flush_list()

            # -- paragraphs ---------------------------------------------------
            new_para = True
            if para:
                last = para[-1]
                last_full = last.x1 >= self.page_cols.get(last.page, (0, 1e9))[1] - 2.0 * self.body
                indented = ln.x0 > left + 0.45 * self.body
                if ln.page == last.page:
                    gap = ln.y0 - last.y1
                    new_para = indented or not last_full or gap > 1.0 * self.body
                else:  # page break: continue if previous line ran to the margin
                    new_para = indented or not last_full or bool(re.search(r"[.!?:\"”)]$", last.text))
                    if not new_para and last.text.endswith("-"):
                        new_para = False
            if new_para:
                flush_para()
            para.append(ln)
            i += 1
        flush_para()
        flush_list()
        return nodes

    # ------------------------------------------------------------- captions
    def _attach_captions(self, nodes: list[Node]) -> list[Node]:
        out: list[Node] = []
        targets = {"Listing": "code", "Example": "code", "Figure": "figure", "Table": "table"}
        used = set()
        for i, n in enumerate(nodes):
            if n.kind != "caption":
                continue
            want = targets.get(n.label, "")
            cap = f"{n.label} {n.number}" + (f" {n.text}" if n.text else "")
            # look forward first for listings/tables (caption above), backward for figures
            order = [1, -1, 2, -2] if want in ("code", "table") else [-1, 1, -2, 2]
            for d in order:
                j = i + d
                if 0 <= j < len(nodes) and nodes[j].kind == want and not nodes[j].caption:
                    nodes[j].caption = cap
                    nodes[j].label, nodes[j].number = n.label, n.number
                    used.add(i)
                    break
        for i, n in enumerate(nodes):
            if n.kind == "caption" and i not in used:
                out.append(Node(kind="para", text=f"{n.label} {n.number} {n.text}".strip(), page=n.page))
            elif n.kind != "caption":
                out.append(n)
        return out

    def _merge_code(self, nodes: list[Node]) -> list[Node]:
        """Join a listing split by a page break (code, code-without-caption)."""
        out: list[Node] = []
        for n in nodes:
            if out and n.kind == "code" == out[-1].kind and not n.caption and n.page != out[-1].page:
                out[-1].text = out[-1].text.rstrip() + "\n" + n.text
                continue
            out.append(n)
        return out


def build_book(results: list[PageResult], title: str = "", author: str = "", spellfix: bool = True) -> Book:
    return StructureBuilder(spellfix=spellfix).build(results, title=title, author=author)
