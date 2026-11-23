"""Synthetic 'bad scan' generator used by the test-suite and benchmarks.

Creates clean text pages with ReportLab, rasterises them with PyMuPDF, then
degrades them like a cheap scanner would: rotation (skew), yellowed paper,
uneven lighting, faded ink and noise. The result is packed into an image-only
PDF, i.e. exactly the kind of input the application is built for.
"""

from __future__ import annotations

import io
import random

import cv2
import pymupdf as fitz
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

WORDS = (
    "the quick brown fox jumps over lazy dog alignment scanner document page "
    "reading order column paragraph table figure research method result data "
    "analysis system design engine quality signal noise image text layout"
).split()


def _sentence(rng: random.Random, n: int) -> str:
    return " ".join(rng.choice(WORDS) for _ in range(n)).capitalize() + "."


def clean_page_pdf(kind: str = "single", seed: int = 0, title: str | None = None) -> bytes:
    """Return a one-page vector PDF. kind: single | two_column | table | figure | blank."""
    rng = random.Random(seed)
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    W, H = A4
    if kind == "blank":
        c.showPage()
        c.save()
        return buf.getvalue()

    c.setFont("Helvetica-Bold", 22)
    c.drawString(60, H - 80, title or "Synthetic Document Title")
    c.setFont("Helvetica", 11)

    if kind == "single":
        y = H - 120
        for _ in range(36):
            c.drawString(60, y, _sentence(rng, 12))
            y -= 18
    elif kind == "two_column":
        for col, x in enumerate((60, W / 2 + 15)):
            y = H - 120
            c.drawString(x, y, f"COLUMN{col + 1}START alpha")
            y -= 16
            for _ in range(34):
                c.drawString(x, y, _sentence(rng, 5)[:44])
                y -= 16
            c.drawString(x, y, f"COLUMN{col + 1}END omega")
    elif kind == "table":
        y = H - 130
        c.drawString(60, y, "Results are summarised in the following table.")
        x0, cw, rh = 60, 120, 24
        rows = [["Name", "Value", "Unit", "Notes"]] + [
            [f"item{i}", str(10 * i), "kg", rng.choice(WORDS)] for i in range(1, 7)
        ]
        top = y - 30
        for r, row in enumerate(rows):
            for k, cell in enumerate(row):
                c.drawString(x0 + k * cw + 6, top - r * rh - 16, cell)
        for r in range(len(rows) + 1):
            c.line(x0, top - r * rh, x0 + 4 * cw, top - r * rh)
        for k in range(5):
            c.line(x0 + k * cw, top, x0 + k * cw, top - len(rows) * rh)
    elif kind == "figure":
        from reportlab.lib.utils import ImageReader
        from PIL import Image

        y = H - 120
        for _ in range(6):
            c.drawString(60, y, _sentence(rng, 12))
            y -= 18
        # A colourful "photo": smooth gradients + circles (continuous tone).
        yy, xx = np.mgrid[0:300, 0:420]
        photo = np.stack([(xx * 255 / 420), (yy * 255 / 300), 255 - (xx * 128 / 420 + yy * 127 / 300)], -1)
        for cx, cy, r in ((120, 150, 70), (300, 110, 50)):
            m = (xx - cx) ** 2 + (yy - cy) ** 2 < r * r
            photo[m] = (240, 200, 40)
        c.drawImage(ImageReader(Image.fromarray(photo.astype(np.uint8))), 90, y - 300, 420, 300)
        y -= 330
        c.drawString(60, y, "Figure 1. A colourful illustration that must stay continuous-tone.")
        y -= 30
        for _ in range(10):
            c.drawString(60, y, _sentence(rng, 12))
            y -= 18
    c.showPage()
    c.save()
    return buf.getvalue()


def rasterize(pdf_bytes: bytes, dpi: int = 200) -> np.ndarray:
    with fitz.open(stream=pdf_bytes, filetype="pdf") as d:
        pix = d[0].get_pixmap(dpi=dpi, colorspace=fitz.csRGB)
        return np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3).copy()


def degrade(img: np.ndarray, angle: float = 0.0, seed: int = 0, rotate90: int = 0) -> np.ndarray:
    """Make a clean raster look like a bad scan."""
    rng = np.random.default_rng(seed)
    h, w = img.shape[:2]
    # Faded ink: lift blacks towards grey.
    out = img.astype(np.float32)
    out = 255 - (255 - out) * 0.65
    # Yellowed paper + vignetting / uneven light.
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    light = 1.0 - 0.18 * (((xx - w * 0.3) / w) ** 2 + ((yy - h * 0.4) / h) ** 2)
    out *= light[..., None]
    out *= np.array([0.97, 0.93, 0.80], np.float32)
    # Skew (content tilt, CCW positive), keeping the same canvas like a scanner.
    if angle:
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        out = cv2.warpAffine(out, M, (w, h), borderValue=(225, 215, 185))
    out += rng.normal(0, 9, out.shape).astype(np.float32)
    out = np.clip(out, 0, 255).astype(np.uint8)
    for _ in range(rotate90 // 90):
        out = cv2.rotate(out, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return out


def images_to_pdf(images: list[np.ndarray], dpi: int = 200) -> bytes:
    doc = fitz.open()
    for img in images:
        h, w = img.shape[:2]
        page = doc.new_page(width=w * 72 / dpi, height=h * 72 / dpi)
        ok, enc = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
        page.insert_image(page.rect, stream=enc.tobytes())
    data = doc.tobytes(garbage=3, deflate=True)
    doc.close()
    return data


def make_scanned_pdf(specs: list[dict], dpi: int = 200) -> bytes:
    """specs: [{"kind": "single", "angle": 3.0, "rotate90": 0, "seed": 1}, ...]"""
    imgs = []
    for i, s in enumerate(specs):
        clean = rasterize(clean_page_pdf(s.get("kind", "single"), seed=s.get("seed", i)), dpi)
        imgs.append(degrade(clean, s.get("angle", 0.0), seed=i, rotate90=s.get("rotate90", 0)))
    return images_to_pdf(imgs, dpi)
