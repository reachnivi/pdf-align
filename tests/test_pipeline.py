"""End-to-end pipeline tests (multi-process, both output modes)."""

import glob
import os
import tempfile

import pymupdf as fitz
import pytest

from pdf_align import PipelineConfig, run_pipeline
from pdf_align.pipeline import PDFInputError


def _pdfalign_tmpdirs():
    return set(glob.glob(os.path.join(tempfile.gettempdir(), "pdfalign_*")))


def test_mode_a_searchable_and_clean(scanned_pdf, tmp_path):
    before = _pdfalign_tmpdirs()
    out = tmp_path / "a.pdf"
    seen = []
    rep = run_pipeline(scanned_pdf, str(out), PipelineConfig(mode="A", dpi=200, workers=2, batch_size=3),
                       progress_cb=lambda done, total, msg, r: seen.append((done, total)))
    assert rep.total == 5 and not rep.errors
    assert [p.index for p in rep.pages] == [0, 1, 2, 3, 4]         # page order preserved
    assert rep.pages[3].status == "blank"
    assert rep.pages[4].orientation_deg == 90
    assert abs(rep.pages[0].skew_deg - (-7.0)) < 0.3
    assert seen[-1][0] == seen[-1][1] == 5                          # progress reached Y of Y
    with fitz.open(out) as doc:
        assert doc.page_count == 5
        assert "Synthetic" in doc[0].get_text()                      # invisible OCR layer
        assert doc[1].search_for("alpha") and doc[1].search_for("omega")
        assert "item3" in doc[2].get_text()
        assert doc[4].rect.height > doc[4].rect.width                # sideways page fixed
    assert _pdfalign_tmpdirs() == before                             # temp files cleaned


def test_mode_b_reflow(scanned_pdf, tmp_path):
    out = tmp_path / "b.pdf"
    rep = run_pipeline(scanned_pdf, str(out), PipelineConfig(mode="B", trim="A4", dpi=200, workers=2))
    assert not rep.errors
    with fitz.open(out) as doc:
        assert abs(doc[0].rect.width - 595.3) < 1
        text = "".join(p.get_text() for p in doc)
    import re

    starts = [m.start() for m in re.finditer(r"COLUMN\S{0,3}START", text)]
    ends = [m.start() for m in re.finditer(r"COLUMN\S{0,3}END", text)]
    assert starts[0] < ends[0] < starts[1] < ends[1]
    assert "Name" in text and "item6" in text


def test_drop_blank_pages(scanned_pdf, tmp_path):
    out = tmp_path / "d.pdf"
    run_pipeline(scanned_pdf, str(out), PipelineConfig(dpi=150, workers=1, ocr=False, blank_policy="drop"))
    with fitz.open(out) as doc:
        assert doc.page_count == 4


def test_rejects_encrypted_pdf(tmp_path):
    src = tmp_path / "enc.pdf"
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "secret")
    doc.save(src, encryption=fitz.PDF_ENCRYPT_AES_256, user_pw="pw", owner_pw="pw")
    with pytest.raises(PDFInputError, match="password"):
        run_pipeline(str(src), str(tmp_path / "o.pdf"))


def test_rejects_non_pdf(tmp_path):
    src = tmp_path / "x.pdf"
    src.write_bytes(b"this is not a pdf")
    with pytest.raises(PDFInputError):
        run_pipeline(str(src), str(tmp_path / "o.pdf"))
