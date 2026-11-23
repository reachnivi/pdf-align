"""Mode B: technical-book reconstruction and typesetting, against ground truth."""

import difflib
import re

import pymupdf as fitz
import pytest

import techbook
from pdf_align import PipelineConfig, deskew, run_pipeline
from pdf_align.layout_engine import Word, grid_text
from pdf_align.structure import build_book
from pdf_align.textfix import fix_line, fix_word, join_lines
from pdf_align.typeset import render_pdf, weasyprint_available


def sim(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


@pytest.fixture(scope="session")
def book_scan(tmp_path_factory):
    d = tmp_path_factory.mktemp("book")
    pdf, gt = techbook.scanned_book(seed=0)
    path = d / "book.pdf"
    path.write_bytes(pdf)
    results = [deskew.process_page(deskew.PageTask(str(path), i, str(d), PipelineConfig()))
               for i in range(gt["pages"])]
    return str(path), gt, results


@pytest.fixture(scope="session")
def book(book_scan):
    _, _, results = book_scan
    return build_book(results, title="Test Book", author="Someone")


# ------------------------------------------------------------------ structure
def test_chapters_and_sections(book, book_scan):
    gt = book_scan[1]
    assert [n.text for n in book.nodes if n.kind == "chapter"] == gt["chapters"]
    assert [n.number for n in book.nodes if n.kind == "chapter"] == ["1", "2"]
    heads = [(n.number, n.text) for n in book.nodes if n.kind == "heading"]
    assert heads == gt["sections"]
    assert all(n.level == 2 for n in book.nodes if n.kind == "heading")


def test_paragraphs_reconstructed(book, book_scan):
    gt = book_scan[1]["paragraphs"]
    paras = [n.text for n in book.nodes if n.kind == "para"]
    assert abs(len(paras) - len(gt)) <= 1, (len(paras), len(gt))
    matched = sum(1 for g in gt if max(sim(g, p) for p in paras) > 0.97)
    assert matched >= len(gt) - 1


def test_running_heads_and_folios_removed(book):
    texts = [n.text for n in book.nodes if n.kind in ("para", "heading")]
    assert not any(re.fullmatch(r"\d+", t.strip()) for t in texts)
    assert not any(t.upper().startswith("CHAPTER") for t in texts)
    # each section heading appears exactly once (recto running heads repeat them)
    joined = " ".join(texts)
    assert joined.count("Configuring the stages") == 1


def test_code_listings_verbatim(book, book_scan):
    gt = book_scan[1]["code"]
    code = [n for n in book.nodes if n.kind == "code"]
    assert len(code) == len(gt)
    for node, truth in zip(code, gt):
        got_lines, gt_lines = node.text.split("\n"), truth.split("\n")
        assert len(got_lines) == len(gt_lines)
        for g, t in zip(got_lines, gt_lines):  # indentation is exact
            assert len(g) - len(g.lstrip()) == len(t) - len(t.lstrip()), (g, t)
        assert sim(node.text, truth) > 0.97, node.text
    assert code[0].text == gt[0]  # the first listing is reproduced exactly


def test_captions_attached(book):
    caps = {n.kind: n.caption for n in book.nodes if n.caption}
    assert caps["code"].startswith("Listing")
    assert caps["table"] == "Table 1.1 Default settings"
    assert caps["figure"] == "Figure 2.1 The three main stages of the engine"
    listing_caps = [n.caption for n in book.nodes if n.kind == "code"]
    assert listing_caps == ["Listing 1.1 A simple processing loop", "Listing 2.1 The pipeline class"]


def test_lists_callouts_tables_figures(book, book_scan):
    gt = book_scan[1]
    lists = [n for n in book.nodes if n.kind == "list"]
    assert [l.ordered for l in lists] == [False, True]
    assert lists[0].items == gt["bullets"]
    assert lists[1].items == gt["numbered"]
    callouts = [n for n in book.nodes if n.kind == "callout"]
    assert len(callouts) == 1 and callouts[0].label == "NOTE" and sim(callouts[0].text, gt["notes"][0]) > 0.97
    table = next(n for n in book.nodes if n.kind == "table")
    assert table.rows[0] == ["Option", "Default", "Meaning"] and table.rows[1] == ["dpi", "300", "Render resolution"]
    assert sum(1 for n in book.nodes if n.kind == "figure") == 1


def test_prose_word_accuracy(book, book_scan):
    gt_words = " ".join(book_scan[1]["paragraphs"]).split()
    out_words = " ".join(n.text for n in book.nodes if n.kind == "para").split()
    sm = difflib.SequenceMatcher(None, gt_words, out_words, autojunk=False)
    errors = sum(max(i2 - i1, j2 - j1) for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag != "equal")
    assert errors / len(gt_words) < 0.01, errors


# ------------------------------------------------------------------ typesetting
needs_weasy = pytest.mark.skipif(not weasyprint_available()[0], reason="WeasyPrint/Pango not installed")


@pytest.fixture(scope="session")
def typeset_pdf(book, tmp_path_factory):
    out = tmp_path_factory.mktemp("ts") / "book.pdf"
    render_pdf(book, str(out))
    return str(out)


@needs_weasy
def test_trim_fonts_and_outline(typeset_pdf):
    with fitz.open(typeset_pdf) as d:
        assert abs(d[0].rect.width - 504) < 0.5 and abs(d[0].rect.height - 661.7) < 0.5   # 7 x 9.19 in
        fonts = {f[3].split("+")[-1] for i in range(d.page_count) for f in d.get_page_fonts(i)}
        assert {"BookSerif", "BookSans-Bold", "BookMono"} <= fonts, fonts
        toc = d.get_toc()
        titles = [t for _, t, _ in toc]
        assert "Getting started with pipelines" in titles and "1.1 Why pipelines matter" in titles


@needs_weasy
def test_toc_page_numbers_and_chapter_openers(typeset_pdf):
    with fitz.open(typeset_pdf) as d:
        for level, title, page in d.get_toc():
            if title == "Contents":
                continue
            plain = title.split(" ", 1)[1] if re.match(r"\d", title) else title
            assert plain in d[page - 1].get_text().replace("\n", " "), (title, page)
            if level == 1:
                assert page % 2 == 1, "chapters start on right-hand (odd) pages"
                top = [b for b in d[page - 1].get_text("blocks") if b[1] < 50]
                assert not top, "no running head on a chapter opener"


@needs_weasy
def test_running_heads_and_code_font(typeset_pdf):
    with fitz.open(typeset_pdf) as d:
        heads = []
        for p in d:
            for b in p.get_text("dict")["blocks"]:
                for l in b.get("lines", []):
                    for s in l["spans"]:
                        if "def process" in s["text"]:
                            assert "BookMono" in s["font"]
                        if s["bbox"][1] < 50:
                            heads.append(s["text"])
        assert any("Chapter 1. Getting started with pipelines" in h for h in heads)
        assert any("1.1 Why pipelines matter" in h or "1.2 Configuring the stages" in h for h in heads)


@needs_weasy
def test_body_text_inside_margins(typeset_pdf):
    with fitz.open(typeset_pdf) as d:
        for p in d:
            for x0, y0, x1, y1, *_ in p.get_text("blocks"):
                assert x0 >= 0.7 * 72 - 1 and x1 <= p.rect.width - 0.7 * 72 + 1, (p.number, x0, x1)


# ------------------------------------------------------------------ end to end + fallback
def test_mode_b_end_to_end_and_fallback(book_scan, tmp_path, monkeypatch):
    path = book_scan[0]
    out = tmp_path / "b.pdf"
    rep = run_pipeline(path, str(out), PipelineConfig(mode="B", workers=2, title="X"))
    assert out.exists() and not rep.errors
    if weasyprint_available()[0]:
        assert rep.warning == ""
    # WeasyPrint missing -> ReportLab fallback still produces a document
    import pdf_align.typeset as ts

    def boom(*a, **k):
        raise OSError("cannot load library 'libpango-1.0-0'")

    monkeypatch.setattr(ts, "render_pdf", boom)
    out2 = tmp_path / "b2.pdf"
    rep2 = run_pipeline(path, str(out2), PipelineConfig(mode="B", workers=1))
    assert out2.exists() and "WeasyPrint unavailable" in rep2.warning
    with fitz.open(out2) as d:
        assert "Getting started" in "".join(p.get_text() for p in d)


# ------------------------------------------------------------------ units
def test_grid_text_rebuilds_indentation_and_glued_tokens():
    pitch, lh = 20, 30

    def w(text, col, row):
        x = 100 + col * pitch
        return Word(text=text, bbox=(x + 2, 50 + row * lh, x + len(text) * pitch - 2, 50 + row * lh + 24))
    words = [w("def", 0, 0), w("f(x):", 4, 0), w("y", 4, 1), w("=", 6, 1), w("g", 8, 1), w("(x)", 9, 1),
             w("return", 4, 3), w("y", 11, 3)]
    assert grid_text(words) == "def f(x):\n    y = g(x)\n\n    return y"


def test_textfix_repairs_ocr_confusions_conservatively():
    assert fix_word("Smal!") == "Small"
    assert fix_word("rnodern") == "modern"
    assert fix_word("wi1l") == "will"
    assert fix_word("Th|s") == "This"
    assert fix_word("pipline", conf=40) == "pipeline"
    for keep in ("json", "HTTP", "x86_64", "estimate_skew(page)", "kubectl", "1.2"):
        assert fix_word(keep, conf=95) == keep
    assert fix_line([("Smal!", 90), ("mistakes", 90)]) == "Small mistakes"
    assert join_lines(["This is infor-", "mation, well-", "Known."]) == "This is information, well- Known."
