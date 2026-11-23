"""Layout analysis / reading-order tests."""

import os
import re

import synth
from pdf_align import deskew as d
from pdf_align.config import PipelineConfig
from pdf_align.layout_engine import xy_cut_order
from pdf_align.models import PageLayout


def _layout(kind, tmp_path, angle=0.0):
    pdf = tmp_path / f"{kind}.pdf"
    pdf.write_bytes(synth.make_scanned_pdf([{"kind": kind, "angle": angle}]))
    res = d.process_page(d.PageTask(str(pdf), 0, str(tmp_path), PipelineConfig()))
    assert res.status == "ok", res.message
    return res.load_layout()


def test_xy_cut_columns_before_rows():
    # title spanning both columns, then 2 columns with aligned paragraph gaps
    boxes = [(0, 0, 100, 10),
             (0, 20, 45, 40), (55, 20, 100, 40),
             (0, 50, 45, 70), (55, 50, 100, 70)]
    assert xy_cut_order(boxes, 5, 3) == [0, 1, 3, 2, 4]


def test_two_column_reading_order(tmp_path):
    lay = _layout("two_column", tmp_path, angle=2.5)
    text = "\n".join(b.text for b in lay.blocks)
    # OCR may read "1" as "I"/"l", so match the markers loosely.
    starts = [m.start() for m in re.finditer(r"COLUMN\S{0,3}START", text)]
    ends = [m.start() for m in re.finditer(r"COLUMN\S{0,3}END", text)]
    assert len(starts) == 2 and len(ends) == 2, text[:400]
    assert text.find("Synthetic") < starts[0] < ends[0] < starts[1] < ends[1], "columns are interleaved"
    assert lay.columns == 2
    assert lay.blocks[0].type == "heading"


def test_ruled_table_is_reconstructed(tmp_path):
    lay = _layout("table", tmp_path, angle=-1.5)
    tables = [b for b in lay.blocks if b.type == "table"]
    assert len(tables) == 1
    cells = tables[0].cells
    assert cells[0] == ["Name", "Value", "Unit", "Notes"]
    assert len(cells) == 7 and all(len(r) == 4 for r in cells)
    assert cells[3][1] == "30"


def test_figure_block_and_crop(tmp_path):
    lay = _layout("figure", tmp_path)
    figs = [b for b in lay.blocks if b.type == "figure"]
    assert figs and figs[0].image_path and os.path.exists(figs[0].image_path)


def test_layout_json_roundtrip(tmp_path):
    lay = _layout("single", tmp_path)
    again = PageLayout.from_json(lay.to_json())
    assert again.to_json() == lay.to_json()
    assert len(again.all_words()) > 100
