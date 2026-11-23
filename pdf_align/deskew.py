"""Computer-vision pipeline: render -> orient -> deskew -> enhance.

Every public function here is a *pure* function on NumPy arrays (or on a PDF
path + page index), which makes them trivially picklable and safe to run in
``ProcessPoolExecutor`` workers. ``process_page`` is the worker entry point:
it receives only a path and a page number, does all the heavy lifting inside
the worker, writes its output to the disk-backed work directory and returns a
tiny ``PageResult`` handle. Page pixels never cross process boundaries.

Conventions
-----------
* Angles are in degrees, counter-clockwise positive (OpenCV convention).
* ``estimate_skew`` returns the angle the *content* is tilted by; to fix it we
  rotate by the negative of that angle.
* Grayscale images are ``uint8`` with white paper = 255.
"""

from __future__ import annotations

import logging
import os
import time
import traceback
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from .config import PipelineConfig
from .models import PageResult

log = logging.getLogger(__name__)

# Pages larger than this many pixels are rendered at a reduced DPI so that a
# single A0 poster inside a book doesn't blow up a worker's memory.
MAX_PIXELS = 60_000_000


class PageRenderError(RuntimeError):
    """Raised when MuPDF cannot rasterise a page (corrupt content stream etc.)."""


# =============================================================================
# Rendering
# =============================================================================

def render_page(pdf_path: str, index: int, dpi: int) -> tuple[np.ndarray, int]:
    """Rasterise a single page to an RGB array.

    The document is opened *inside* this call so that each worker process has
    its own MuPDF handle and only one page is ever resident in memory.

    Returns ``(rgb_image, effective_dpi)``. The effective DPI can be lower
    than requested for gigantic pages (see ``MAX_PIXELS``).
    """
    import pymupdf as fitz  # imported lazily so the module imports fast

    try:
        with fitz.open(pdf_path) as doc:
            page = doc.load_page(index)
            rect = page.rect
            w_in, h_in = rect.width / 72.0, rect.height / 72.0
            eff_dpi = dpi
            if w_in * h_in * dpi * dpi > MAX_PIXELS:
                eff_dpi = int((MAX_PIXELS / max(w_in * h_in, 1e-6)) ** 0.5)
                log.warning("page %d is very large; rendering at %d dpi", index + 1, eff_dpi)
            # get_pixmap honours /Rotate, so mixed portrait/landscape pages come
            # out the way a viewer would show them.
            pix = page.get_pixmap(dpi=eff_dpi, colorspace=fitz.csRGB, alpha=False)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3).copy()
            del pix
    except Exception as exc:  # noqa: BLE001 - MuPDF raises a zoo of types
        raise PageRenderError(f"cannot render page {index + 1}: {exc}") from exc
    if img.size == 0:
        raise PageRenderError(f"page {index + 1} rendered to an empty image")
    return img, eff_dpi


def to_gray(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def _downscale(gray: np.ndarray, target_long: int) -> tuple[np.ndarray, float]:
    h, w = gray.shape[:2]
    scale = min(1.0, target_long / float(max(h, w)))
    if scale < 1.0:
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return gray, scale


# =============================================================================
# Background normalisation & blank detection
# =============================================================================

def normalize_background(gray: np.ndarray) -> np.ndarray:
    """Flatten uneven illumination / yellowed paper to a white background.

    Estimates the paper background with a large morphological *closing*
    (which removes dark text strokes) on a 1/4-scale copy, then divides the
    image by it. Text keeps its relative darkness; paper becomes ~255.
    """
    h, w = gray.shape
    small = cv2.resize(gray, (max(1, w // 4), max(1, h // 4)), interpolation=cv2.INTER_AREA)
    k = max(5, int(min(small.shape) / 60) | 1)
    bg = cv2.morphologyEx(small, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    bg = cv2.medianBlur(bg, k if k <= 255 else 255)
    bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR)
    np.maximum(bg, 1, out=bg)
    # uint8 division with saturation: no full-size float temporaries.
    return cv2.divide(gray, bg, scale=255.0)


def ink_ratio(gray: np.ndarray, margin: float = 0.05) -> float:
    """Fraction of 'ink' pixels, ignoring a margin where scanner shadows live."""
    small, _ = _downscale(gray, 1000)
    norm = normalize_background(small)
    h, w = norm.shape
    mh, mw = int(h * margin), int(w * margin)
    core = norm[mh:h - mh, mw:w - mw] if h > 2 * mh and w > 2 * mw else norm
    return float(np.mean(core < 150))


def is_blank(gray: np.ndarray, threshold: float = 0.0015) -> bool:
    """A page is blank when < 0.15% of its (inner) area is ink.

    Dust specks and faint bleed-through stay below this; a single short line
    of text at 300 DPI is already above it.
    """
    return ink_ratio(gray) < threshold


# =============================================================================
# Orientation (0 / 90 / 180 / 270)
# =============================================================================

def detect_orientation(gray: np.ndarray) -> int:
    """Return the clockwise rotation (0/90/180/270) needed to make text upright.

    Two questions, answered by the tool best suited to each:
      1. *Sideways or not?*  Geometry: are text lines horizontal or vertical
         (skew-tolerant projection profiles of glyph-sized components).
      2. *Up or down?*  Tesseract OSD, asked on a copy that has already been
         turned upright-or-upside-down, deskewed and stripped of ruling lines
         (the conditions under which OSD is reliable).
    Without Tesseract only step 1 is applied.
    """
    small, _ = _downscale(gray, 2400)
    sideways, ratio = _sideways_score(small)

    # Grid-like pages (tables) make the geometry ambiguous: then let OSD judge
    # both hypotheses and keep the one it is happiest with.
    candidates = [sideways] if ratio >= 2.0 else [sideways, not sideways]
    best = None  # (score, base_rot, rotate, conf)
    for sw in candidates:
        base = cv2.rotate(small, cv2.ROTATE_90_CLOCKWISE) if sw else small
        angle, _ = estimate_skew(base)
        if angle:
            base = rotate_bound(base, -angle)
        rotate, conf = _osd(_clean_for_osd(base))
        base_rot = 90 if sw else 0
        # Prefer the geometric answer unless OSD is clearly happier elsewhere.
        score = conf - (0.0 if sw == sideways else 1.0) - (3.0 if rotate in (90, 270) else 0.0)
        if best is None or score > best[0]:
            best = (score, base_rot, rotate, conf)
    _, base_rot, rotate, conf = best

    if rotate in (90, 270):
        # OSD disagrees with geometry; only believe it when it is very sure
        # and the geometric evidence was weak.
        if conf >= 4.0 and ratio < 1.6:
            return (base_rot + rotate) % 360
        return base_rot
    # A wrong 180 flip is far worse than leaving a page as-is: require some
    # confidence.
    flip = rotate == 180 and conf >= 1.0
    return (base_rot + (180 if flip else 0)) % 360


def _osd(clean: np.ndarray) -> tuple[int, float]:
    """Tesseract OSD -> (clockwise rotation to fix, confidence). (0, 0) if unavailable."""
    try:
        import pytesseract
    except ImportError:
        return 0, 0.0
    for min_chars in (10, 4):
        try:
            osd = pytesseract.image_to_osd(
                clean, config=f"--psm 0 -c min_characters_to_try={min_chars}",
                output_type=pytesseract.Output.DICT, timeout=30,
            )
            return int(osd.get("rotate", 0)) % 360, float(osd.get("orientation_conf", 0.0))
        except pytesseract.TesseractError:
            continue  # "too few characters": retry with a lower bar
        except Exception as exc:  # noqa: BLE001 - tesseract binary missing, timeout
            log.debug("OSD unavailable: %s", exc)
            break
    return 0, 0.0


def _clean_for_osd(gray_small: np.ndarray) -> np.ndarray:
    """Binarised, denoised copy with table rules / boxes removed."""
    clean = cv2.medianBlur(normalize_background(gray_small), 3)
    _, clean = cv2.threshold(clean, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    inv = 255 - clean
    h, w = inv.shape
    for kernel in ((max(15, w // 30), 1), (1, max(15, h // 30))):
        lines = cv2.morphologyEx(inv, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, kernel))
        clean[cv2.dilate(lines, np.ones((3, 3), np.uint8)) > 0] = 255
    return clean


def _sideways_score(gray_small: np.ndarray) -> tuple[bool, float]:
    """Are text lines vertical (page rotated ~90 degrees)?

    Uses character-sized components only (rules / borders removed) and
    compares the sharpest *column* profile against the sharpest *row* profile
    over the skew search range, so it also works on skewed pages. Returns
    (sideways, ratio) where ratio is the winner's margin.
    """
    xs, ys = _ink_points(gray_small)
    if len(xs) < 2000:
        return False, 1.0
    angles = np.arange(-15, 15.1, 1.0)
    rows = max(_profile_sharpness(xs, ys, a) for a in angles)
    cols = max(_profile_sharpness(ys, -xs, a) for a in angles)  # 90-degree turned
    if cols > 1.1 * rows:
        return True, cols / max(rows, 1e-9)
    return False, rows / max(cols, 1e-9)


def apply_orientation(img: np.ndarray, rotate_cw: int) -> np.ndarray:
    if rotate_cw == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if rotate_cw == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    if rotate_cw == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


# =============================================================================
# Skew estimation
# =============================================================================

def _ink_points(gray_small: np.ndarray, max_points: int = 250_000) -> tuple[np.ndarray, np.ndarray]:
    # Median blur first: uniform sensor noise surviving binarisation would
    # create an artificial histogram step at the (zeroed) margins at 0 deg.
    norm = cv2.medianBlur(normalize_background(gray_small), 3)
    _, binary = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    # Ignore the outer 3% where scan borders / punch holes produce strong,
    # misleading straight edges.
    h, w = binary.shape
    mh, mw = int(h * 0.03), int(w * 0.03)
    binary[:mh, :] = 0
    binary[h - mh:, :] = 0
    binary[:, :mw] = 0
    binary[:, w - mw:] = 0
    # Keep only character-sized components. Scanner-bed edges, page borders,
    # long table rules and big pictures are aligned to the *scanner* axes (or
    # nothing at all) and would otherwise create a false peak at 0 degrees.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n > 1:
        cw, ch, area = stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT], stats[:, cv2.CC_STAT_AREA]
        keep = (ch >= 3) & (ch <= h * 0.06) & (cw <= w * 0.12) & (area >= 8)
        keep[0] = False
        if keep.sum() >= 30:  # enough glyphs; otherwise fall back to all ink
            binary = np.where(keep, 255, 0).astype(np.uint8)[labels]
    ys, xs = np.nonzero(binary)
    if len(xs) > max_points:
        sel = np.random.default_rng(0).choice(len(xs), max_points, replace=False)
        xs, ys = xs[sel], ys[sel]
    jitter = np.random.default_rng(1).uniform(-0.5, 0.5, size=(2, len(xs))).astype(np.float32)
    return xs.astype(np.float32) + jitter[0], ys.astype(np.float32) + jitter[1]


def _profile_sharpness(xs: np.ndarray, ys: np.ndarray, angle_deg: float) -> float:
    """Sharpness of the horizontal projection after rotating points by -angle.

    When text lines are perfectly horizontal the row histogram alternates
    between dense (text line) and empty (leading) bins, which maximises the
    sum of squared differences between neighbouring bins (Postl's criterion).
    """
    t = np.deg2rad(angle_deg)
    # Rotating content by -angle maps a point's y to:  y*cos(t) + x*sin(t)
    # (image y axis points down, so CCW-positive tilt raises the right side).
    # Points carry sub-pixel jitter (see _ink_points) and are binned at exactly
    # 1 px, which avoids quantisation artefacts that favour 0 degrees.
    yp = ys * np.cos(t) + xs * np.sin(t)
    hist = np.bincount((yp - yp.min()).astype(np.int64))
    d = np.diff(hist.astype(np.float64))
    return float(np.dot(d, d))


def _hough_skew(gray_small: np.ndarray, max_deg: float) -> Optional[float]:
    """Median angle of long, near-horizontal line segments (text baselines,
    table rules). Returns None if not enough evidence."""
    norm = normalize_background(gray_small)
    _, binary = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    # Smear characters horizontally so each text line becomes a solid bar.
    kw = max(9, gray_small.shape[1] // 60)
    smeared = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 1)))
    edges = cv2.Canny(smeared, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 1800, threshold=80,
                            minLineLength=gray_small.shape[1] // 6, maxLineGap=kw)
    if lines is None or len(lines) < 5:
        return None
    angles = []
    for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):  # shape differs across OpenCV versions
        a = -np.degrees(np.arctan2(y2 - y1, x2 - x1))  # CCW positive
        if abs(a) <= max_deg:
            angles.append(a)
    if len(angles) < 5:
        return None
    return float(np.median(angles))


def estimate_skew(gray: np.ndarray, max_deg: float = 15.0) -> tuple[float, float]:
    """Estimate the page skew in degrees (CCW positive) and a confidence score.

    Hybrid strategy:
      1. Projection-profile sweep over +/-max_deg in 0.5 degree steps (robust, global).
      2. Hough transform on smeared text lines as an independent estimate.
      3. Fine profile search in 0.05 degree steps around the winning coarse angle.

    Confidence is the ratio of the best sharpness to the median sharpness of
    the sweep: ~1.0 means "no dominant line direction" (photos, blank pages).
    """
    small, _ = _downscale(gray, 1400)
    xs, ys = _ink_points(small)
    if len(xs) < 2000:
        return 0.0, 0.0

    coarse = np.arange(-max_deg, max_deg + 1e-6, 0.5)
    scores = np.array([_profile_sharpness(xs, ys, a) for a in coarse])
    best_idx = int(np.argmax(scores))
    confidence = float(scores[best_idx] / max(np.median(scores), 1e-9))
    best = float(coarse[best_idx])

    hough = _hough_skew(small, max_deg)
    if hough is not None and abs(hough - best) > 1.0:
        # Disagreement: pick whichever is sharper by the profile criterion.
        if _profile_sharpness(xs, ys, hough) > scores[best_idx]:
            best = hough

    fine = np.arange(best - 0.6, best + 0.6 + 1e-6, 0.05)
    fscores = [_profile_sharpness(xs, ys, a) for a in fine]
    best = float(fine[int(np.argmax(fscores))])

    if confidence < 1.15:
        # No dominant text direction - rotating would only make things worse.
        return 0.0, confidence
    return round(best, 2), confidence


# =============================================================================
# Rotation without cropping
# =============================================================================

def rotate_bound(img: np.ndarray, angle_deg: float, border_value=None) -> np.ndarray:
    """Rotate by ``angle_deg`` (CCW positive) on an enlarged canvas.

    The output canvas is computed from the rotated corner positions so that
    *no* part of the original page (text, borders, stamps) is ever cropped.
    New corner areas are filled with the page's paper colour.
    """
    if abs(angle_deg) < 0.01:
        return img
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw = int(np.ceil(h * sin + w * cos))
    nh = int(np.ceil(h * cos + w * sin))
    M[0, 2] += nw / 2.0 - cx
    M[1, 2] += nh / 2.0 - cy
    if border_value is None:
        border_value = paper_color(img)
    return cv2.warpAffine(img, M, (nw, nh), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=border_value)


def paper_color(img: np.ndarray):
    """Estimate paper colour as the 90th percentile brightness (per channel)."""
    sample = img[::8, ::8]
    if sample.ndim == 2:
        return int(np.percentile(sample, 90))
    return tuple(int(v) for v in np.percentile(sample.reshape(-1, sample.shape[-1]), 90, axis=0))


def trim_to_original(rgb: np.ndarray, gray: np.ndarray, orig_hw: tuple[int, int]):
    """Crop the enlarged rotation canvas back towards the original page size.

    ``rotate_bound`` grows the canvas so nothing can be cut off. Usually the
    extra area is just paper, so we cut a window of the original size centred
    on the content - but the window is grown whenever content would be lost,
    so text and page borders are never cropped.
    """
    h, w = gray.shape
    h0, w0 = orig_hw
    if h <= h0 and w <= w0:
        return rgb, gray
    small, sc = _downscale(gray, 1000)
    ink = normalize_background(small) < 160
    ys, xs = np.nonzero(ink)
    if len(xs) < 50:
        cx0, cy0, cx1, cy1 = 0, 0, w, h
    else:
        cx0, cx1 = np.percentile(xs, [0.1, 99.9]) / sc
        cy0, cy1 = np.percentile(ys, [0.1, 99.9]) / sc
    margin = 0.02 * max(h, w)
    ww = int(min(w, max(w0, (cx1 - cx0) + 2 * margin)))
    hh = int(min(h, max(h0, (cy1 - cy0) + 2 * margin)))
    x = int(np.clip((cx0 + cx1) / 2 - ww / 2, 0, w - ww))
    y = int(np.clip((cy0 + cy1) / 2 - hh / 2, 0, h - hh))
    # Copies, so the larger rotated buffers can be freed.
    return rgb[y:y + hh, x:x + ww].copy(), gray[y:y + hh, x:x + ww].copy()


# =============================================================================
# Enhancement
# =============================================================================

@dataclass
class EnhanceResult:
    image: np.ndarray                 # uint8 gray (H,W) or RGB (H,W,3)
    ink: np.ndarray                   # uint8 binary mask, ink = 255
    figure_boxes: list[tuple[int, int, int, int]] = field(default_factory=list)
    bilevel: bool = True              # True if `image` is pure black & white


def sauvola(gray: np.ndarray, window: int, k: float = 0.2, R: float = 128.0) -> np.ndarray:
    """Sauvola local thresholding (returns ink mask, ink=255).

    Unlike a global threshold it adapts to local contrast, which keeps faded
    strokes and doesn't fill in dark regions. Implemented with box filters,
    so it's O(N) regardless of window size.
    """
    g = gray.astype(np.float32)
    mean = cv2.boxFilter(g, cv2.CV_32F, (window, window), borderType=cv2.BORDER_REFLECT)
    sq = cv2.sqrBoxFilter(g, cv2.CV_32F, (window, window), borderType=cv2.BORDER_REFLECT)
    # Everything below is in place to keep peak memory at ~3 float planes.
    np.subtract(sq, mean * mean, out=sq)
    np.maximum(sq, 0, out=sq)
    np.sqrt(sq, out=sq)                      # sq is now the local std
    sq *= k / R
    sq += 1.0 - k
    sq *= mean                               # sq is now the threshold
    del mean
    return cv2.compare(g, sq, cv2.CMP_LT)    # uint8 0/255


def _remove_specks(ink: np.ndarray, min_area: int) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    if n <= 1:
        return ink
    lut = np.where(stats[:, cv2.CC_STAT_AREA] >= min_area, 255, 0).astype(np.uint8)
    lut[0] = 0
    return lut[labels]  # uint8 lookup: avoids a large int64 temporary


def _remove_scan_borders(ink: np.ndarray, band: float = 0.06) -> np.ndarray:
    """Erase dark scanner frames / shadows: components touching the image edge
    whose bounding box lies entirely inside the outer margin band."""
    h, w = ink.shape
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    bh, bw = int(h * band), int(w * band)
    kill = np.zeros(n, dtype=bool)
    for i in range(1, n):
        x, y, cw, ch, _ = stats[i]
        touches = x == 0 or y == 0 or x + cw >= w or y + ch >= h
        if not touches:
            continue
        in_band = (cw <= bw) or (ch <= bh)  # a strip along one edge
        long_strip = cw > w * 0.3 or ch > h * 0.3
        if in_band and long_strip:
            kill[i] = True
    if kill.any():
        lut = np.where(kill, 0, 255).astype(np.uint8)
        lut[0] = 0
        ink = cv2.bitwise_and(ink, lut[labels])
    return ink


def global_normalize(gray: np.ndarray) -> np.ndarray:
    """Scale so the paper level (95th percentile) maps to white.

    Unlike ``normalize_background`` this is a *global* operation, so large
    dark/continuous-tone regions (photos) keep their tones instead of being
    mistaken for background and divided away.
    """
    paper = float(np.percentile(gray[::4, ::4], 95)) or 255.0
    return np.clip(gray.astype(np.float32) * (255.0 / paper), 0, 255).astype(np.uint8)


def find_figure_regions(gnorm: np.ndarray, ink: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Locate photos / halftones / diagrams that must NOT be binarised.

    ``gnorm`` must be *globally* normalised (see ``global_normalize``).
    Two cues:
      * density of non-paper pixels over a large window (photos, shaded
        charts): body text covers ~10-20% of its area, pictures > 45%;
      * very large connected ink components (line drawings).
    """
    h, w = gnorm.shape
    page_area = float(h * w)
    scale = 0.25
    small_norm = cv2.resize(gnorm, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    mid = (small_norm < 200).astype(np.float32)
    k = max(9, int(min(small_norm.shape) / 25) | 1)
    density = cv2.blur(mid, (k, k))
    mask = (density > 0.45).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))

    boxes: list[tuple[int, int, int, int]] = []
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for i in range(1, n):
        x, y, cw, ch, area = stats[i]
        if area / (mask.shape[0] * mask.shape[1]) > 0.01:
            boxes.append((int(x / scale), int(y / scale), int((x + cw) / scale), int((y + ch) / scale)))

    # Large line-art components (diagrams). Tables also produce these; the
    # layout engine removes boxes that turn out to be ruled tables.
    small_ink = cv2.resize(ink, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)
    n, _, stats, _ = cv2.connectedComponentsWithStats(small_ink, connectivity=8)
    for i in range(1, n):
        x, y, cw, ch, area = stats[i]
        box_area = (cw / scale) * (ch / scale)
        if box_area / page_area > 0.04 and cw > 0.15 * mask.shape[1] and ch > 0.08 * mask.shape[0]:
            boxes.append((int(x / scale), int(y / scale), int((x + cw) / scale), int((y + ch) / scale)))

    return _merge_boxes(boxes, pad=int(0.005 * max(h, w)), limit=(w, h))


def _box_overlap(a, b) -> float:
    """Intersection area over the smaller box's area."""
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return ix * iy / max(small, 1)


def _merge_boxes(boxes, pad: int, limit) -> list[tuple[int, int, int, int]]:
    """Pad boxes and merge any that overlap, until stable."""
    W, H = limit
    boxes = [(max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad), min(H, y1 + pad)) for x0, y0, x1, y1 in boxes]
    changed = True
    while changed:
        changed = False
        out: list[tuple[int, int, int, int]] = []
        for b in boxes:
            for i, o in enumerate(out):
                if b[0] <= o[2] and o[0] <= b[2] and b[1] <= o[3] and o[1] <= b[3]:
                    out[i] = (min(b[0], o[0]), min(b[1], o[1]), max(b[2], o[2]), max(b[3], o[3]))
                    changed = True
                    break
            else:
                out.append(b)
        boxes = out
    return boxes


def enhance(rgb: np.ndarray, gray: np.ndarray, dpi: int, cfg: PipelineConfig) -> EnhanceResult:
    """Whiten paper, denoise, sharpen faded text, preserve images.

    Text regions -> background normalisation, denoise, levels stretch, Sauvola
    binarisation, speck removal (crisp black-on-white).
    Figure regions -> background normalisation + CLAHE (in colour when the
    source is colourful), so photos and diagrams are not posterised.
    """
    s = cfg.enhance_strength
    norm = normalize_background(gray)

    # Edge-preserving denoise. Non-local means is the best quality option but
    # its cost grows with resolution, so it is run at <=200 DPI-equivalent and
    # the result upsampled; text edges are re-sharpened by the threshold step.
    h_param = int(round(5 * s))
    if h_param > 0:
        factor = min(1.0, 200.0 / max(dpi, 1))
        if factor < 1.0:
            small = cv2.resize(norm, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
            den = cv2.fastNlMeansDenoising(small, None, h=h_param, templateWindowSize=7, searchWindowSize=15)
            den = cv2.resize(den, (norm.shape[1], norm.shape[0]), interpolation=cv2.INTER_CUBIC)
            # Keep full-res detail: only take the low-frequency part from the
            # denoised copy where the original is near paper colour.
            norm = np.where(norm > 200, np.maximum(norm, den), np.minimum(norm, den)).astype(np.uint8)
        else:
            norm = cv2.fastNlMeansDenoising(norm, None, h=h_param, templateWindowSize=7, searchWindowSize=15)

    # Levels stretch: paper -> pure white, faded grey strokes -> darker.
    ink_px = norm[norm < 200]
    lo = float(np.percentile(ink_px, 5)) if ink_px.size > 500 else 0.0
    hi = 235.0 - 10.0 * (s - 1.0)
    lo = min(lo, hi - 60)
    # Levels + gamma as a 256-entry lookup table (no full-size float images).
    # The gamma > 1 darkens mid-greys (faded text) more than near-black.
    levels = np.clip((np.arange(256, dtype=np.float32) - lo) * (255.0 / (hi - lo)), 0, 255)
    lut = (255.0 * (levels / 255.0) ** (1.0 + 0.35 * s)).astype(np.uint8)
    stretched = cv2.LUT(norm, lut)

    window = max(15, int(dpi / 10) | 1)
    ink = sauvola(stretched, window, k=0.18 + 0.04 * (s - 1.0))
    ink[stretched > 245] = 0          # never ink on near-white paper
    ink[stretched < 60] = 255         # always ink on very dark strokes
    ink = _remove_scan_borders(ink)
    ink = _remove_specks(ink, min_area=max(3, int(6 * (dpi / 300.0) ** 2)))

    gnorm = global_normalize(gray)
    figures = find_figure_regions(gnorm, ink)
    if figures:
        # Ruled tables form big line-art components too, but they are text and
        # must be binarised / re-typeset, not kept as pictures.
        from .layout_engine import detect_ruled_tables

        tables = [t["bbox"] for t in detect_ruled_tables(ink, dpi)]
        figures = [f for f in figures if not any(_box_overlap(f, t) > 0.5 for t in tables)]

    if not cfg.binarize_text:
        out = stretched.copy()
        out[(ink == 0) & (stretched > 200)] = 255
        bilevel = False
    else:
        out = 255 - ink
        bilevel = not figures

    if figures:
        colourful = False
        if rgb.ndim == 3:
            # Measure saturation on non-paper pixels only: yellowed paper is
            # "colourful" but must not make a B/W photo keep its tint.
            sats = []
            for x0, y0, x1, y1 in figures:
                content = gnorm[y0:y1, x0:x1] < 200
                if content.any():
                    sat = cv2.cvtColor(rgb[y0:y1, x0:x1], cv2.COLOR_RGB2HSV)[..., 1]
                    sats.append(float(sat[content].mean()))
            colourful = bool(sats) and float(np.mean(sats)) > 45
        clahe = cv2.createCLAHE(clipLimit=1.5 + 0.5 * s, tileGridSize=(8, 8))
        if colourful:
            out = cv2.cvtColor(out, cv2.COLOR_GRAY2RGB) if out.ndim == 2 else out
            for x0, y0, x1, y1 in figures:
                region = cv2.cvtColor(rgb[y0:y1, x0:x1], cv2.COLOR_RGB2LAB)
                L = clahe.apply(global_normalize(region[..., 0]))
                region[..., 0] = L
                rgb_region = cv2.cvtColor(region, cv2.COLOR_LAB2RGB)
                rgb_region[L > 235] = 255  # whiten paper inside the figure
                out[y0:y1, x0:x1] = rgb_region
        else:
            for x0, y0, x1, y1 in figures:
                out[y0:y1, x0:x1] = clahe.apply(gnorm[y0:y1, x0:x1])
        bilevel = False

    return EnhanceResult(image=out, ink=ink, figure_boxes=figures, bilevel=bilevel)


# =============================================================================
# Saving
# =============================================================================

def save_page_image(img: np.ndarray, path_no_ext: str, bilevel: bool, jpeg_quality: int) -> str:
    """Persist the processed page as compactly as possible.

    * Pure black & white pages -> 1-bit PNG (tiny, razor-sharp text).
    * Pages with photos -> JPEG at ``jpeg_quality`` (or PNG if quality == 0).
    """
    from PIL import Image

    if bilevel:
        path = path_no_ext + ".png"
        Image.fromarray(img if img.ndim == 2 else to_gray(img)).convert("1").save(path, optimize=True)
        return path
    pil = Image.fromarray(img)
    if jpeg_quality <= 0:
        path = path_no_ext + ".png"
        pil.save(path, optimize=True)
    else:
        path = path_no_ext + ".jpg"
        pil.save(path, quality=jpeg_quality, optimize=True, subsampling=0 if img.ndim == 2 else 2)
    return path


# =============================================================================
# Worker entry point
# =============================================================================

@dataclass(frozen=True)
class PageTask:
    pdf_path: str
    index: int
    work_dir: str
    config: PipelineConfig


def worker_init() -> None:
    """Process-pool initializer: avoid thread oversubscription.

    Each worker already owns one core; letting OpenCV and Tesseract (OpenMP)
    spin up their own thread pools per process would thrash the CPU.
    """
    os.environ.setdefault("OMP_THREAD_LIMIT", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    cv2.setNumThreads(1)


def process_page(task: PageTask) -> PageResult:
    """Full per-page pipeline. Never raises: failures become ``status='error'``.

    render -> blank check -> orientation -> skew -> rotate -> enhance
           -> OCR + layout analysis -> persist image + layout JSON
    """
    from . import layout_engine  # local import keeps worker start-up light

    t0 = time.perf_counter()
    cfg = task.config
    stem = os.path.join(task.work_dir, f"page_{task.index:05d}")
    res = PageResult(index=task.index, dpi=cfg.dpi)

    # 1. Render ------------------------------------------------------------
    try:
        rgb, dpi = render_page(task.pdf_path, task.index, cfg.dpi)
    except PageRenderError as exc:
        res.status, res.message = "error", str(exc)
        res.seconds = time.perf_counter() - t0
        return res
    res.dpi = dpi
    gray = to_gray(rgb)

    try:
        # 2. Blank page ------------------------------------------------------
        if is_blank(gray):
            res.status = "blank"
            res.width_px, res.height_px = gray.shape[1], gray.shape[0]
            res.seconds = time.perf_counter() - t0
            return res

        # 3. Orientation (mixed portrait / landscape / upside-down scans) ----
        if cfg.detect_orientation:
            rot = detect_orientation(gray)
            if rot:
                rgb, gray = apply_orientation(rgb, rot), apply_orientation(gray, rot)
            res.orientation_deg = rot

        # 4. Skew ------------------------------------------------------------
        angle, _conf = estimate_skew(gray, cfg.max_skew_deg)
        res.skew_deg = angle
        if angle:
            orig_hw = gray.shape[:2]
            rgb = rotate_bound(rgb, -angle)
            gray = to_gray(rgb)  # cheaper than a second cubic warp
            rgb, gray = trim_to_original(rgb, gray, orig_hw)

        # 5. Enhance ---------------------------------------------------------
        if cfg.enhance:
            enh = enhance(rgb, gray, dpi, cfg)
        else:
            norm = normalize_background(gray)
            _, ink = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
            enh = EnhanceResult(image=rgb, ink=ink, figure_boxes=[], bilevel=False)

        res.height_px, res.width_px = enh.image.shape[:2]
        res.image_path = save_page_image(enh.image, stem, enh.bilevel, cfg.jpeg_quality)

        # 6. OCR + layout ----------------------------------------------------
        layout = layout_engine.build_layout(enh, dpi, cfg, crop_prefix=stem)
        res.layout_path = stem + ".json"
        with open(res.layout_path, "w", encoding="utf-8") as fh:
            fh.write(layout.to_json())

    except Exception as exc:  # noqa: BLE001 - one bad page must not kill the job
        log.exception("page %d failed", task.index + 1)
        res.status = "error"
        res.message = f"{type(exc).__name__}: {exc}"
        # Fallback: keep the original rendering so the output has no hole.
        try:
            res.height_px, res.width_px = rgb.shape[:2]
            res.image_path = save_page_image(rgb, stem + "_raw", False, cfg.jpeg_quality or 85)
        except Exception:  # noqa: BLE001
            res.image_path = None
            res.message += " | " + traceback.format_exc(limit=1)

    res.seconds = time.perf_counter() - t0
    return res
