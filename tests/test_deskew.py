"""Computer-vision unit tests on synthetic degraded scans."""

import cv2
import numpy as np
import pytest

import synth
from pdf_align import deskew as d
from pdf_align.config import PipelineConfig


def _scan(kind="single", angle=0.0, rotate90=0, dpi=200):
    return synth.degrade(synth.rasterize(synth.clean_page_pdf(kind), dpi), angle, rotate90=rotate90)


@pytest.mark.parametrize("angle", [-12.0, -7.0, -2.3, 0.0, 0.8, 3.5, 9.0])
def test_skew_estimate_is_accurate(angle):
    est, conf = d.estimate_skew(d.to_gray(_scan(angle=angle)))
    assert abs(est - angle) <= 0.3, (angle, est, conf)


def test_skew_on_two_column_and_table():
    for kind, angle in (("two_column", 4.0), ("table", -1.5)):
        est, _ = d.estimate_skew(d.to_gray(_scan(kind, angle)))
        assert abs(est - angle) <= 0.3


def test_skew_on_blank_page_is_zero():
    est, conf = d.estimate_skew(d.to_gray(_scan("blank")))
    assert est == 0.0


def test_rotate_bound_never_crops():
    img = np.full((400, 300), 255, np.uint8)
    img[0:10, :] = 0           # ink touching all four edges
    img[-10:, :] = 0
    img[:, 0:10] = 0
    img[:, -10:] = 0
    ink_before = int((img < 128).sum())
    out = d.rotate_bound(img, 10.0, border_value=255)
    assert out.shape[0] > 400 and out.shape[1] > 300
    ink_after = int((out < 128).sum())
    assert ink_after >= 0.97 * ink_before  # only interpolation loss


def test_trim_keeps_all_content():
    img = _scan(angle=6.0)
    gray = d.to_gray(img)
    rot_rgb, rot_gray = d.rotate_bound(img, -6.0), d.rotate_bound(gray, -6.0)
    t_rgb, t_gray = d.trim_to_original(rot_rgb, rot_gray, gray.shape)
    assert t_gray.shape[0] >= gray.shape[0] * 0.99 and t_gray.shape[1] >= gray.shape[1] * 0.99
    ink = lambda g: int((d.normalize_background(g) < 128).sum())
    assert ink(t_gray) >= 0.98 * ink(rot_gray)


def test_orientation_detects_sideways_and_upside_down():
    assert d.detect_orientation(d.to_gray(_scan(rotate90=90))) == 90
    upside = cv2.rotate(_scan(), cv2.ROTATE_180)
    assert d.detect_orientation(d.to_gray(upside)) == 180
    assert d.detect_orientation(d.to_gray(_scan())) == 0


def test_blank_detection():
    assert d.is_blank(d.to_gray(_scan("blank")))
    assert not d.is_blank(d.to_gray(_scan("single")))


def test_enhance_whitens_paper_and_keeps_text():
    img = _scan()
    enh = d.enhance(img, d.to_gray(img), 200, PipelineConfig())
    out = enh.image
    assert enh.bilevel and out.ndim == 2
    paper = out[out.shape[0] // 2 - 200:, :60]  # left margin strip
    assert paper.mean() > 250
    assert (out < 128).mean() > 0.01  # text survived


def test_enhance_preserves_photos():
    img = synth.degrade(synth.rasterize(synth.clean_page_pdf("figure"), 200), 0)
    enh = d.enhance(img, d.to_gray(img), 200, PipelineConfig())
    assert enh.figure_boxes, "photo not detected"
    assert not enh.bilevel
    x0, y0, x1, y1 = enh.figure_boxes[0]
    region = enh.image[y0:y1, x0:x1]
    # continuous tone: many distinct levels, not just black & white
    assert len(np.unique(region)) > 50


def test_process_page_handles_corrupt_input(tmp_path):
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"%PDF-1.4\n garbage")
    res = d.process_page(d.PageTask(str(bad), 0, str(tmp_path), PipelineConfig()))
    assert res.status == "error" and res.message


@pytest.mark.parametrize("kind", ["single", "table", "two_column"])
@pytest.mark.parametrize("turn", [0, 90, 180, 270])
def test_orientation_all_rotations(kind, turn):
    """Pages scanned in any of the four orientations come back upright -
    including grid-like table pages where line geometry alone is ambiguous."""
    codes = {90: cv2.ROTATE_90_COUNTERCLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_CLOCKWISE}
    g = d.to_gray(_scan(kind, angle=2.0, dpi=300))
    if turn:
        g = cv2.rotate(g, codes[turn])
    assert d.detect_orientation(g) == turn
