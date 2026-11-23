"""OCR + document layout analysis + reading-order reconstruction.

Pipeline (runs inside the worker, on the already deskewed/enhanced page):

    OCR words  ──►  lines  ──►  text blocks (paragraph candidates)
                                       │
    ruled-table detection (OpenCV) ────┤  words inside tables become cells
    figure regions (from deskew) ──────┤  text inside figures leaves the flow
                                       ▼
                 recursive XY-cut reading order (column aware)
                                       ▼
          classification: heading / paragraph / list / table / figure /
                          header / footer
                                       ▼
                               PageLayout (JSON)

OCR engines are pluggable through the small ``OCREngine`` protocol. Tesseract
is the default (fast, CPU-only, gives word boxes + block/paragraph/line
structure). Surya is supported as an optional, heavier engine.
"""

from __future__ import annotations

import logging
import re
import statistics
from typing import Optional, Protocol

import cv2
import numpy as np

from .config import PipelineConfig
from .models import Block, Line, PageLayout, Word

log = logging.getLogger(__name__)

HEADER_BAND = 0.07   # top/bottom fraction of the page treated as header/footer
BULLET_RE = re.compile(r"^\s*([•●▪–\-\*•·o]|\(?\d{1,3}[.)]|\(?[a-zA-Z][.)])\s+")


# =============================================================================
# OCR engines
# =============================================================================

class OCREngine(Protocol):
    name: str

    def recognize(self, gray: np.ndarray, lang: str) -> list[Word]:
        """Return words with pixel bboxes in ``gray``'s coordinate system.
        ``block``/``par``/``line`` ids group words when the engine knows them."""


class TesseractEngine:
    name = "tesseract"

    def recognize(self, gray: np.ndarray, lang: str) -> list[Word]:
        import pytesseract

        # --psm 1: automatic page segmentation with OSD-free layout analysis;
        # preserve_interword_spaces keeps column gaps measurable.
        data = pytesseract.image_to_data(
            gray, lang=lang, config="--oem 1 --psm 1 -c preserve_interword_spaces=1",
            output_type=pytesseract.Output.DICT, timeout=240,
        )
        words: list[Word] = []
        for i, txt in enumerate(data["text"]):
            txt = (txt or "").strip()
            conf = float(data["conf"][i])
            if not txt or conf < 0:
                continue
            # Discard low-confidence single glyphs: typically noise or rules.
            if conf < 25 and len(txt) <= 2 and not txt.isalnum():
                continue
            x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
            words.append(Word(text=txt, bbox=(x, y, x + w, y + h), conf=conf,
                              block=data["block_num"][i], par=data["par_num"][i], line=data["line_num"][i]))
        return words


class SuryaEngine:
    """Optional transformer OCR (``pip install surya-ocr``).

    Surya returns *lines* rather than words; each line becomes one ``Word``
    spanning the line, which is fine both for the invisible text layer and for
    reflow. Block structure is inferred geometrically afterwards. Models are
    loaded once per worker process and cached.
    """

    name = "surya"
    _predictors = None

    def _load(self):
        if SuryaEngine._predictors is None:
            from surya.detection import DetectionPredictor  # type: ignore
            from surya.recognition import RecognitionPredictor  # type: ignore

            try:  # surya >= 0.14 needs a shared foundation model
                from surya.foundation import FoundationPredictor  # type: ignore

                rec = RecognitionPredictor(FoundationPredictor())
            except ImportError:
                rec = RecognitionPredictor()
            SuryaEngine._predictors = (DetectionPredictor(), rec)
        return SuryaEngine._predictors

    def recognize(self, gray: np.ndarray, lang: str) -> list[Word]:
        from PIL import Image

        det, rec = self._load()
        img = Image.fromarray(gray).convert("RGB")
        try:
            preds = rec([img], det_predictor=det)
        except TypeError:  # older API also wanted languages
            preds = rec([img], [[lang[:2]]], det)
        words: list[Word] = []
        for li, tl in enumerate(preds[0].text_lines):
            if not tl.text.strip():
                continue
            x0, y0, x1, y1 = tl.bbox
            words.append(Word(text=tl.text.strip(), bbox=(x0, y0, x1, y1),
                              conf=100.0 * float(getattr(tl, "confidence", 1.0) or 1.0),
                              block=0, par=0, line=li + 1))
        return words


def get_engine(name: str) -> OCREngine:
    if name == "surya":
        try:
            import surya  # noqa: F401

            return SuryaEngine()
        except ImportError:
            log.warning("surya-ocr not installed; falling back to Tesseract")
    return TesseractEngine()


def tesseract_available() -> tuple[bool, str]:
    try:
        import pytesseract

        return True, str(pytesseract.get_tesseract_version())
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


# =============================================================================
# Geometry helpers
# =============================================================================

def _union(boxes) -> tuple[float, float, float, float]:
    xs0, ys0, xs1, ys1 = zip(*boxes)
    return (min(xs0), min(ys0), max(xs1), max(ys1))


def _center_in(b, region) -> bool:
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    return region[0] <= cx <= region[2] and region[1] <= cy <= region[3]


def _overlap_frac(a, b) -> float:
    """Fraction of a's area covered by b."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area = max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))
    return ix * iy / area


# =============================================================================
# Words -> lines -> blocks
# =============================================================================

def words_to_lines(words: list[Word]) -> list[Line]:
    """Group words into lines using the engine's (block, par, line) ids, then
    split a line wherever a horizontal gap is so large it must be a column
    gutter (Tesseract occasionally bridges narrow gutters)."""
    groups: dict[tuple, list[Word]] = {}
    for w in words:
        groups.setdefault((w.block, w.par, w.line), []).append(w)
    lines: list[Line] = []
    for ws in groups.values():
        ws.sort(key=lambda w: w.bbox[0])
        h = statistics.median([w.bbox[3] - w.bbox[1] for w in ws])
        cur = [ws[0]]
        for w in ws[1:]:
            if w.bbox[0] - cur[-1].bbox[2] > 3.0 * h:
                lines.append(Line(words=cur, bbox=_union([x.bbox for x in cur])))
                cur = []
            cur.append(w)
        lines.append(Line(words=cur, bbox=_union([x.bbox for x in cur])))
    return lines


def lines_to_blocks(lines: list[Line], words_have_structure: bool) -> list[list[Line]]:
    """Cluster lines into paragraph-like blocks.

    With Tesseract we start from its (block, par) grouping and only split it
    where line spacing jumps. Without structure (Surya) lines are merged
    geometrically: same column (x-overlap) and small vertical gap.
    """
    if not lines:
        return []
    med_h = statistics.median([l.height for l in lines]) or 1.0

    if words_have_structure:
        groups: dict[tuple, list[Line]] = {}
        for l in lines:
            w0 = l.words[0]
            # column-split pieces of one tesseract line get distinct x-bucket
            groups.setdefault((w0.block, w0.par, int(l.bbox[0] // (med_h * 8))), []).append(l)
        # Re-merge buckets of the same paragraph whose lines are vertically
        # interleaved with left-aligned starts (the x-bucket is only a guard).
        raw = list(groups.values())
    else:
        raw = [[l] for l in lines]

    # Geometric pass: sort, then merge consecutive blocks that are clearly the
    # same paragraph (or split blocks with an abnormal gap).
    out: list[list[Line]] = []
    for blk in raw:
        blk.sort(key=lambda l: l.bbox[1])
        cur = [blk[0]]
        for l in blk[1:]:
            gap = l.bbox[1] - cur[-1].bbox[3]
            if gap > 1.6 * med_h:
                out.append(cur)
                cur = []
            cur.append(l)
        out.append(cur)

    if not words_have_structure:
        out.sort(key=lambda b: (b[0].bbox[1], b[0].bbox[0]))
        merged: list[list[Line]] = []
        for blk in out:
            bb = _union([l.bbox for l in blk])
            for m in merged:
                mb = _union([l.bbox for l in m])
                xov = min(bb[2], mb[2]) - max(bb[0], mb[0])
                if xov > 0.5 * min(bb[2] - bb[0], mb[2] - mb[0]) and 0 <= bb[1] - mb[3] < 0.9 * med_h:
                    m.extend(blk)
                    break
            else:
                merged.append(blk)
        out = merged
    return out


# =============================================================================
# Table detection
# =============================================================================

def _cluster_positions(values: list[int], tol: int) -> list[int]:
    if not values:
        return []
    values = sorted(values)
    clusters = [[values[0]]]
    for v in values[1:]:
        if v - clusters[-1][-1] <= tol:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    return [int(np.mean(c)) for c in clusters]


def detect_rules(ink: np.ndarray, dpi: int = 300) -> tuple[np.ndarray, np.ndarray]:
    """Horizontal and vertical ruling-line masks (ink = 255).

    A rule must be *long* (longer than any glyph: stems of big bold chapter
    titles are ~0.3 in tall) and *thin* (thick bars are text or graphics).
    """
    h, w = ink.shape
    hk = max(20, w // 35, int(dpi * 0.45))
    vk = max(20, h // 50, int(dpi * 0.45))
    horiz = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (hk, 1)))
    vert = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, vk)))
    max_thick = max(3, dpi // 30)
    for mask, axis in ((horiz, 0), (vert, 1)):
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        thick = stats[:, cv2.CC_STAT_HEIGHT if axis == 0 else cv2.CC_STAT_WIDTH]
        bad = thick > max_thick
        bad[0] = False
        if bad.any():
            mask[bad[labels]] = 0
    return horiz, vert


def detect_ruled_tables(ink: np.ndarray, dpi: int, rules=None) -> list[dict]:
    """Find tables drawn with ruling lines.

    Morphological opening with long horizontal / vertical kernels keeps only
    rules; their union forms the grid. Each grid region with >= 2 rows and
    >= 2 columns becomes a table with explicit row/column boundaries.
    """
    h, w = ink.shape
    horiz, vert = rules if rules is not None else detect_rules(ink, dpi)
    grid = cv2.dilate(cv2.bitwise_or(horiz, vert), np.ones((3, 3), np.uint8), iterations=2)

    tables = []
    n, _, stats, _ = cv2.connectedComponentsWithStats(grid, connectivity=8)
    min_w, min_h = w * 0.15, dpi * 0.25
    tol = max(4, dpi // 40)
    for i in range(1, n):
        x, y, cw, ch, _ = stats[i]
        if cw < min_w or ch < min_h:
            continue
        sub_h = horiz[y:y + ch, x:x + cw]
        sub_v = vert[y:y + ch, x:x + cw]
        # A rule must span most of the table; tolerate small breaks/residual skew
        # by testing a few-pixel-thick band.
        band = max(3, dpi // 100)
        rows_y = np.nonzero((cv2.dilate(sub_h, np.ones((band, 1), np.uint8)) > 0).sum(axis=1) > 0.6 * cw)[0]
        cols_x = np.nonzero((cv2.dilate(sub_v, np.ones((1, band), np.uint8)) > 0).sum(axis=0) > 0.6 * ch)[0]
        rows = _cluster_positions([int(v) + y for v in rows_y], tol)
        cols = _cluster_positions([int(v) + x for v in cols_x], tol)
        # Tables with only outer border or without inner vertical rules:
        # fall back to column inference from word positions later.
        if len(rows) < 3:
            continue
        tables.append({"bbox": (x, y, x + cw, y + ch), "rows": rows, "cols": cols})
    return tables


def _infer_columns_from_words(words: list[Word], x0: float, x1: float, min_gap: float) -> list[float]:
    """Column boundaries from vertical whitespace runs shared by all rows."""
    width = int(x1 - x0) + 1
    occ = np.zeros(width, dtype=bool)
    for wd in words:
        occ[max(0, int(wd.bbox[0] - x0)):min(width, int(wd.bbox[2] - x0) + 1)] = True
    bounds = [x0]
    run_start = None
    for i, filled in enumerate(occ):
        if not filled and run_start is None:
            run_start = i
        elif filled and run_start is not None:
            if i - run_start >= min_gap and run_start > 0:
                bounds.append(x0 + (run_start + i) / 2)
            run_start = None
    bounds.append(x1)
    return bounds


def build_table_cells(table: dict, words: list[Word], med_h: float) -> list[list[str]]:
    x0, y0, x1, y1 = table["bbox"]
    rows = table["rows"] if len(table["rows"]) >= 2 else [y0, y1]
    cols = table["cols"]
    inner = [w for w in words if _center_in(w.bbox, table["bbox"])]
    if len(cols) < 3:  # no inner vertical rules -> infer from whitespace
        cols = _infer_columns_from_words(inner, x0, x1, min_gap=1.2 * med_h)
    grid = [["" for _ in range(len(cols) - 1)] for _ in range(len(rows) - 1)]
    # reading order inside cells: by text line (tolerant to jitter), then x
    for wd in sorted(inner, key=lambda w: (round(((w.bbox[1] + w.bbox[3]) / 2) / max(med_h * 0.7, 1)),
                                           w.bbox[0])):
        cx, cy = (wd.bbox[0] + wd.bbox[2]) / 2, (wd.bbox[1] + wd.bbox[3]) / 2
        r = int(np.searchsorted(rows, cy)) - 1
        c = int(np.searchsorted(cols, cx)) - 1
        if 0 <= r < len(grid) and 0 <= c < len(grid[0]):
            grid[r][c] = (grid[r][c] + " " + wd.text).strip()
    # drop fully empty rows/cols (double rules, header spacing)
    grid = [row for row in grid if any(cell for cell in row)]
    if grid:
        keep = [j for j in range(len(grid[0])) if any(row[j] for row in grid)]
        grid = [[row[j] for j in keep] for row in grid]
    return grid


def detect_borderless_table(blk: list[Line], med_h: float) -> Optional[list[list[str]]]:
    """A block is a borderless table if >= 3 of its lines split into >= 2
    cells at large gaps, with consistent cell counts and aligned starts."""
    if len(blk) < 3:
        return None
    rows: list[list[tuple[float, str]]] = []
    for l in blk:
        cells, cur = [], [l.words[0]]
        for w in l.words[1:]:
            if w.bbox[0] - cur[-1].bbox[2] > 1.8 * med_h:
                cells.append(cur)
                cur = []
            cur.append(w)
        cells.append(cur)
        rows.append([(c[0].bbox[0], " ".join(w.text for w in c)) for c in cells])
    counts = [len(r) for r in rows]
    mode = max(set(counts), key=counts.count)
    # A bullet / number column "aligns" too, but that's a list, not a table.
    if all(len(r[0][1]) <= 3 for r in rows if len(r) >= 2):
        return None
    if mode < 2 or counts.count(mode) < max(3, int(0.7 * len(rows))):
        return None
    starts = np.array([[x for x, _ in r] for r in rows if len(r) == mode])
    if np.max(np.std(starts, axis=0)) > 2.5 * med_h:  # columns not aligned
        return None
    return [[t for _, t in r] + [""] * (mode - len(r)) if len(r) <= mode else
            [t for _, t in r[:mode - 1]] + [" ".join(t for _, t in r[mode - 1:])] for r in rows]



# =============================================================================
# Frames (shaded panels / ruled boxes) and code detection
# =============================================================================

CODE_SYMBOLS = set("{}()[]=<>:;_/\\\"'#|&*+")
CODE_RE = re.compile(r"(\bdef\b|\bclass\b|\breturn\b|\bimport\b|\bfunction\b|\bvar\b|\bconst\b|"
                     r"\bpublic\b|\bprivate\b|\bif\s*\(|=>|==|!=|\+\+|::|->|\w+\(\S*\)|^\s*[{}]\s*$|;\s*$|:\s*$)")


def detect_panels(gray: Optional[np.ndarray], dpi: int) -> list[tuple[float, float, float, float]]:
    """Light-grey filled rectangles (code listing / sidebar backgrounds)."""
    if gray is None:
        return []
    from .deskew import find_tinted_regions

    w = gray.shape[1]  # noqa: F841
    from .deskew import is_panel

    return [tuple(map(float, b)) for b, fill, sd in find_tinted_regions(gray, min_area=0.004)
            if is_panel(b, fill, w, dpi, sd)]


def detect_rule_boxes(ink: np.ndarray, dpi: int, rules=None) -> list[tuple[float, float, float, float]]:
    """Closed rectangles drawn with rules that are *not* tables (callouts,
    sidebars): exactly two long horizontal rules joined by vertical rules."""
    h, w = ink.shape
    horiz, vert = rules if rules is not None else detect_rules(ink, dpi)
    grid = cv2.dilate(cv2.bitwise_or(horiz, vert), np.ones((3, 3), np.uint8), iterations=2)
    out = []
    n, _, stats, _ = cv2.connectedComponentsWithStats(grid, connectivity=8)
    band = max(3, dpi // 100)
    tol = max(4, dpi // 40)
    for i in range(1, n):
        x, y, cw, ch, _ = stats[i]
        if cw < w * 0.3 or ch < dpi * 0.2:
            continue
        sub_h = cv2.dilate(horiz[y:y + ch, x:x + cw], np.ones((band, 1), np.uint8)) > 0
        sub_v = cv2.dilate(vert[y:y + ch, x:x + cw], np.ones((1, band), np.uint8)) > 0
        rows = _cluster_positions(list(np.nonzero(sub_h.sum(axis=1) > 0.6 * cw)[0]), tol)
        cols = _cluster_positions(list(np.nonzero(sub_v.sum(axis=0) > 0.6 * ch)[0]), tol)
        if len(rows) == 2 and len(cols) >= 2:
            out.append((float(x), float(y), float(x + cw), float(y + ch)))
    return out


def grid_fit(words: list[Word]) -> float:
    """How well word starts sit on a fixed character grid (0 = perfect, ~0.25 =
    random). Monospaced code scores low; proportional prose scores high."""
    ws = [w for w in words if len(w.text) >= 3]
    if len(ws) < 6:
        return 1.0
    pitch = float(np.median([(w.bbox[2] - w.bbox[0]) / (len(w.text) - 0.2) for w in ws]))
    if pitch <= 0:
        return 1.0
    left = min(w.bbox[0] for w in words)
    fr = [((w.bbox[0] - left) / pitch) % 1.0 for w in words]
    return float(np.mean([min(f, 1 - f) for f in fr]))


def code_score(lines: list[Line]) -> float:
    """Heuristic 0..1 likelihood that a block of lines is source code."""
    text = "\n".join(l.text for l in lines)
    if not text.strip():
        return 0.0
    sym = sum(1 for c in text if c in CODE_SYMBOLS) / max(1, len(text.replace(" ", "")))
    hits = sum(1 for l in lines if CODE_RE.search(l.text))
    prose = sum(1 for l in lines if re.search(r"[a-z]{3,} [a-z]{3,} [a-z]{3,} [a-z]{3,}", l.text)
                and not CODE_RE.search(l.text))
    fit = grid_fit([w for l in lines for w in l.words])
    score = 0.0
    score += min(sym / 0.08, 1.0) * 0.45
    score += (hits / len(lines)) * 0.35
    score += (0.2 if fit < 0.12 else 0.0)
    score -= (prose / len(lines)) * 0.4
    return max(0.0, min(1.0, score))


def reocr_code(gray: np.ndarray, bbox, lang: str) -> Optional[str]:
    """Re-recognise a code region and rebuild it on the monospace grid.

    Uses Tesseract without dictionaries (identifiers are not English words)
    and reconstructs indentation and intra-line spacing from word positions,
    because Tesseract drops leading spaces and splits tokens such as
    ``estimate_skew(page)``.
    """
    import pytesseract

    h, w = gray.shape[:2]
    pad = 6
    x0, y0, x1, y1 = (int(max(0, bbox[0] - pad)), int(max(0, bbox[1] - pad)),
                      int(min(w, bbox[2] + pad)), int(min(h, bbox[3] + pad)))
    crop = gray[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    cfg = ("--oem 1 --psm 6 -c preserve_interword_spaces=1 "
           "-c load_system_dawg=0 -c load_freq_dawg=0")
    try:
        d = pytesseract.image_to_data(crop, lang=lang, config=cfg, output_type=pytesseract.Output.DICT, timeout=60)
    except Exception as exc:  # noqa: BLE001
        log.debug("code re-OCR failed: %s", exc)
        return None
    words = [Word(text=t.strip(), bbox=(d["left"][i], d["top"][i], d["left"][i] + d["width"][i],
                                        d["top"][i] + d["height"][i]),
                  conf=float(d["conf"][i]), block=d["block_num"][i], par=d["par_num"][i], line=d["line_num"][i])
             for i, t in enumerate(d["text"]) if t and t.strip() and float(d["conf"][i]) >= 0]
    return grid_text(words)


def _fit_pitch(starts: np.ndarray, guess: float) -> float:
    """Character pitch that puts the most word starts on an integer grid."""
    best, best_err = guess, 1e9
    for p in np.linspace(0.85 * guess, 1.15 * guess, 121):
        fr = (starts / p) % 1.0
        err = float(np.mean(np.minimum(fr, 1 - fr)))
        if err < best_err:
            best, best_err = float(p), err
    return best


def grid_text(words: list[Word]) -> Optional[str]:
    """Lay words out on a monospace character grid -> verbatim code text.

    The pitch is fitted so that word *start* positions land on integer
    columns; a word whose column equals the current line length is glued to
    the previous token (e.g. ``estimate_skew`` + ``(page)``).
    """
    if not words:
        return None
    longish = [w for w in words if len(w.text) >= 3] or words
    guess = float(np.median([(w.bbox[2] - w.bbox[0]) / max(len(w.text) - 0.2, 0.8) for w in longish]))
    guess = max(guess, 1.0)
    left = min(w.bbox[0] for w in words)
    pitch = _fit_pitch(np.array([w.bbox[0] - left for w in words], dtype=np.float64), guess)
    # group into visual lines by vertical centre
    words = sorted(words, key=lambda w: ((w.bbox[1] + w.bbox[3]) / 2, w.bbox[0]))
    line_h = float(np.median([w.bbox[3] - w.bbox[1] for w in words]))
    rows: list[list[Word]] = []
    for wd in words:
        cy = (wd.bbox[1] + wd.bbox[3]) / 2
        if rows and abs(cy - np.mean([(x.bbox[1] + x.bbox[3]) / 2 for x in rows[-1]])) < 0.5 * line_h:
            rows[-1].append(wd)
        else:
            rows.append([wd])
    centers = [float(np.mean([(x.bbox[1] + x.bbox[3]) / 2 for x in r])) for r in rows]
    steps = np.diff(centers)
    pitch_y = float(np.min(steps[steps > 0.5 * line_h])) if len(steps) and (steps > 0.5 * line_h).any() \
        else line_h * 1.3
    out_lines: list[str] = []
    for k, r in enumerate(rows):
        if k:
            out_lines.extend([""] * max(0, int(round((centers[k] - centers[k - 1]) / pitch_y)) - 1))
        r.sort(key=lambda w: w.bbox[0])
        sline, prev_x1 = "", None
        for wd in r:
            col = int(round((wd.bbox[0] - left) / pitch))
            if sline and col == len(sline):
                sline += wd.text                     # exactly adjacent on the grid
            elif sline and col < len(sline):
                # Grid disrupted (OCR added a stray glyph earlier on the line):
                # fall back to the real pixel gap to the previous word.
                sline += (" " if wd.bbox[0] - prev_x1 > 0.6 * pitch else "") + wd.text
            else:
                sline += " " * (col - len(sline)) + wd.text
            prev_x1 = wd.bbox[2]
        out_lines.append(sline.rstrip())
    return fix_code_text("\n".join(out_lines))


def fix_code_text(text: str) -> str:
    """Undo typical OCR artefacts in code without touching its meaning."""
    table = {"\u201c": '"', "\u201d": '"', "\u201e": '"', "\u2018": "'", "\u2019": "'", "\u2032": "'",
             "\u2014": "--", "\u2013": "-", "\u2122": "", "\u00ae": "", "\u00a9": "", "\u00bb": ">>",
             "\u00ab": "<<", "\u2026": "...", "\u00a0": " "}
    for k, v in table.items():
        text = text.replace(k, v)
    # Unbalanced opening brace right after another opener ("[{render, ...]"):
    # a classic OCR insertion - drop it.
    fixed = []
    for line in text.split("\n"):
        if line.count("{") > line.count("}") and not line.rstrip().endswith("{"):
            line = re.sub(r"([\[(])\{", r"\1", line)
        fixed.append(line)
    return "\n".join(fixed)


def line_ink(ink: Optional[np.ndarray], bbox) -> float:
    if ink is None:
        return 0.0
    x0, y0, x1, y1 = (int(v) for v in bbox)
    region = ink[max(0, y0):max(0, y1), max(0, x0):max(0, x1)]
    return float(region.mean() / 255.0) if region.size else 0.0

# =============================================================================
# Reading order: recursive XY-cut
# =============================================================================

def xy_cut_order(boxes: list[tuple], min_gap_x: float, min_gap_y: float) -> list[int]:
    """Return indices of ``boxes`` in human reading order.

    Recursive XY-cut, *vertical cuts first*: if a whitespace gutter runs the
    full height of a region, everything left of it is read before anything
    right of it (columns). Otherwise the region is split at horizontal gaps
    (bands: title, column section, footer...) and each band is recursed.
    This correctly handles full-width headings above multi-column text.
    """

    def gaps(intervals, lo, hi, min_gap):
        """(start, end) of empty runs >= min_gap strictly inside [lo, hi]."""
        intervals = sorted(intervals)
        out, reach = [], lo
        for a, b in intervals:
            if a - reach >= min_gap and reach > lo:
                out.append((reach, a))
            reach = max(reach, b)
        return out

    def split(idx, cuts, axis):
        lo_k, hi_k = (0, 2) if axis == "x" else (1, 3)
        edges = [-1e18] + sorted(cuts) + [1e18]
        parts = [[i for i in idx if edges[k] <= (boxes[i][lo_k] + boxes[i][hi_k]) / 2 < edges[k + 1]]
                 for k in range(len(edges) - 1)]
        return [p for p in parts if p]

    def coverage_gutters(idx, x0, x1, height):
        """x-ranges covered by < 30% of the peak coverage but flanked by
        well-covered text on both sides: column gutters that are merely
        interrupted by a few spanning blocks."""
        if height <= 0 or x1 - x0 < 3 * min_gap_x:
            return []
        off = int(x0)
        occ = np.zeros(int(x1) - off + 2, np.float32)
        for i in idx:
            occ[int(boxes[i][0]) - off:int(boxes[i][2]) - off + 1] += boxes[i][3] - boxes[i][1]
        occ /= height
        low = occ < 0.3 * max(float(occ.max()), 1e-6)
        out, start = [], None
        for k, v in enumerate(low):
            if v and start is None:
                start = k
            elif not v and start is not None:
                if k - start >= min_gap_x and start > 0:
                    out.append((start + off, k + off))
                start = None
        return out

    def rec(idx: list[int]) -> list[int]:
        if len(idx) <= 1:
            return idx
        x0 = min(boxes[i][0] for i in idx)
        x1 = max(boxes[i][2] for i in idx)
        y0 = min(boxes[i][1] for i in idx)
        y1 = max(boxes[i][3] for i in idx)

        # 1) A gutter running the full height of the region: read everything
        #    on its left first (columns).
        vg = gaps([(boxes[i][0], boxes[i][2]) for i in idx], x0, x1, min_gap_x)
        if vg:
            parts = split(idx, [(a + b) / 2 for a, b in vg], "x")
            if len(parts) > 1:
                return [j for p in parts for j in rec(p)]

        # 2) Horizontal bands. If a column gutter is visible in the coverage
        #    that is only interrupted by spanning blocks (titles, wide figures),
        #    cut only next to those spanning blocks. Cutting at every gap would
        #    slice the columns wherever their paragraph breaks happen to line
        #    up, interleaving the columns.
        hg = gaps([(boxes[i][1], boxes[i][3]) for i in idx], y0, y1, min_gap_y)
        if hg:
            gutters = coverage_gutters(idx, x0, x1, y1 - y0)
            if gutters:
                spanning = [i for i in idx
                            if any(boxes[i][0] < (ga + gb) / 2 < boxes[i][2] for ga, gb in gutters)]
                eps = min_gap_y
                allowed = [(a, b) for a, b in hg
                           if any(abs(boxes[i][3] - a) <= eps or abs(boxes[i][1] - b) <= eps for i in spanning)]
            else:
                allowed = hg
            parts = split(idx, [(a + b) / 2 for a, b in allowed], "y") if allowed else []
            if len(parts) > 1:
                return [j for p in parts for j in rec(p)]

        # No clean cut (overlapping boxes): top-to-bottom, left-to-right.
        return sorted(idx, key=lambda i: (boxes[i][1], boxes[i][0]))

    return rec(list(range(len(boxes))))


def count_columns(blocks: list[Block], page_w: float) -> int:
    body = [b for b in blocks if b.type in ("paragraph", "list")]
    if not body:
        return 1
    narrow = [b for b in body if (b.bbox[2] - b.bbox[0]) < 0.6 * page_w]
    if len(narrow) < 2:
        return 1
    centers = sorted((b.bbox[0] + b.bbox[2]) / 2 for b in narrow)
    return 1 + sum(1 for a, b in zip(centers, centers[1:]) if b - a > 0.25 * page_w)


# =============================================================================
# Main entry point
# =============================================================================

def analyze_layout(words: list[Word], page_w: int, page_h: int, dpi: int,
                   ink: Optional[np.ndarray] = None,
                   figure_boxes: Optional[list[tuple]] = None,
                   has_structure: bool = True,
                   tables: Optional[list[dict]] = None,
                   frames: Optional[list[tuple[tuple, str]]] = None) -> PageLayout:
    """Turn OCR words (+ optional CV cues) into an ordered, typed PageLayout."""
    layout = PageLayout(width=page_w, height=page_h, ocr_words=list(words), dpi=dpi)
    frames = list(frames or [])
    if not words and not figure_boxes:
        return layout

    heights = [w.bbox[3] - w.bbox[1] for w in words if len(w.text) > 1] or [dpi / 6]
    med_h = float(statistics.median(heights))
    layout.median_line_height = med_h

    # --- tables -----------------------------------------------------------
    if tables is None:
        tables = detect_ruled_tables(ink, dpi) if ink is not None else []
    table_blocks: list[Block] = []
    for t in tables:
        cells = build_table_cells(t, words, med_h)
        if cells and len(cells) >= 2 and len(cells[0]) >= 2:
            table_blocks.append(Block(type="table", bbox=tuple(map(float, t["bbox"])), cells=cells))
    table_regions = [b.bbox for b in table_blocks]

    # --- figures (drop those that are actually tables) --------------------
    figures = [tuple(map(float, f)) for f in (figure_boxes or [])
               if not any(_overlap_frac(f, t) > 0.5 or _overlap_frac(t, f) > 0.5 for t in table_regions)]
    # A "figure" full of well-recognised words is a shaded text panel (code
    # listing, sidebar) rather than a picture: let its text flow.
    kept = []
    for f in figures:
        inside = [w for w in words if _center_in(w.bbox, f) and w.conf >= 60 and len(w.text) >= 2]
        area = max(1.0, (f[2] - f[0]) * (f[3] - f[1]))
        cover = sum((w.bbox[2] - w.bbox[0]) * (w.bbox[3] - w.bbox[1]) for w in inside) / area
        if len(inside) >= 8 and cover > 0.10:
            frames.append((f, "shaded"))
        else:
            kept.append(f)
    figures = kept
    figure_blocks = [Block(type="figure", bbox=f) for f in figures]

    # --- text flow: words not consumed by tables / figures ----------------
    flow_words = [w for w in words
                  if not any(_center_in(w.bbox, r) for r in table_regions)
                  and not any(_center_in(w.bbox, f) for f in figures)]
    lines = words_to_lines(flow_words)
    text_blocks: list[Block] = []
    for blk in lines_to_blocks(lines, has_structure):
        bbox = _union([l.bbox for l in blk])
        cells = detect_borderless_table(blk, med_h)
        if cells:
            text_blocks.append(Block(type="table", bbox=bbox, cells=cells, lines=blk))
        else:
            text_blocks.append(Block(type="paragraph", bbox=bbox, lines=blk))

    # --- frames & code ----------------------------------------------------
    for b in text_blocks:
        for fb, kind in frames:
            if _center_in(b.bbox, fb):
                b.frame = kind
                break
    for b in text_blocks:
        if b.type != "paragraph":
            continue
        score = code_score(b.lines)
        if score >= 0.5 or (b.frame == "shaded" and score >= 0.25):
            b.type = "code"
    # Code split by Tesseract into several blocks (blank lines inside a
    # listing) is merged back: same frame, or vertically adjacent code.
    text_blocks.sort(key=lambda b: (b.bbox[1], b.bbox[0]))
    merged: list[Block] = []
    for b in text_blocks:
        prev = merged[-1] if merged else None
        if (prev is not None and b.type == prev.type == "code"
                and (b.frame and b.frame == prev.frame or b.bbox[1] - prev.bbox[3] < 3 * med_h)):
            prev.lines.extend(b.lines)
            prev.bbox = _union([prev.bbox, b.bbox])
            continue
        # A code panel's lines that scored as prose (e.g. a lone "class X:")
        if (prev is not None and prev.type == "code" and b.frame == "shaded" and prev.frame == "shaded"
                and b.bbox[1] - prev.bbox[3] < 3 * med_h):
            prev.lines.extend(b.lines)
            prev.bbox = _union([prev.bbox, b.bbox])
            continue
        merged.append(b)
    # ...and a prose-scored block directly above code in the same panel.
    for i in range(len(merged) - 1):
        a, b = merged[i], merged[i + 1]
        if a.type == "paragraph" and b.type == "code" and a.frame == b.frame == "shaded":
            a.type = "code"
    text_blocks = []
    for b in merged:
        if text_blocks and b.type == text_blocks[-1].type == "code" and b.frame == text_blocks[-1].frame:
            text_blocks[-1].lines = b.lines + text_blocks[-1].lines if b.bbox[1] < text_blocks[-1].bbox[1] \
                else text_blocks[-1].lines + b.lines
            text_blocks[-1].bbox = _union([text_blocks[-1].bbox, b.bbox])
        else:
            text_blocks.append(b)
    for b in text_blocks:
        for l in b.lines:
            l.ink = line_ink(ink, l.bbox)

    blocks = text_blocks + table_blocks + figure_blocks

    # --- classification ---------------------------------------------------
    for b in blocks:
        if b.type != "paragraph" or b.frame:
            continue
        n_lines = len(b.lines)
        line_h = statistics.median([l.height for l in b.lines])
        text = b.text.strip()
        in_top = b.bbox[3] < page_h * HEADER_BAND
        in_bottom = b.bbox[1] > page_h * (1 - HEADER_BAND)
        if (in_top or in_bottom) and n_lines <= 2 and len(text) < 120:
            b.type = "header" if in_top else "footer"
            continue
        ratio = line_h / med_h if med_h else 1.0
        short = n_lines <= 3 and len(text) < 140 and not text.endswith((".", ",", ";"))
        if short and ratio >= 1.3:
            b.type = "heading"
            b.level = 1 if ratio >= 1.8 else 2
        elif n_lines <= 2 and len(text) < 90 and text.isupper() and len(text) > 3:
            b.type = "heading"
            b.level = 3
        elif n_lines >= 1 and BULLET_RE.match(b.lines[0].text) and \
                sum(1 for l in b.lines if BULLET_RE.match(l.text)) >= max(1, n_lines // 3):
            b.type = "list"

    # --- reading order ----------------------------------------------------
    order = xy_cut_order([b.bbox for b in blocks], min_gap_x=max(0.9 * med_h, dpi * 0.08),
                         min_gap_y=0.5 * med_h)
    ordered = [blocks[i] for i in order]
    # Headers first and footers last regardless of geometry.
    layout.blocks = ([b for b in ordered if b.type == "header"]
                     + [b for b in ordered if b.type not in ("header", "footer")]
                     + [b for b in ordered if b.type == "footer"])
    layout.columns = count_columns(layout.blocks, page_w)
    return layout


def build_layout(enh, dpi: int, cfg: PipelineConfig, crop_prefix: str) -> PageLayout:
    """Worker-side glue: OCR the enhanced page, analyse it, save figure crops.

    ``enh`` is a ``deskew.EnhanceResult``. All coordinates in the returned
    layout are in the pixel space of ``enh.image``.
    """
    img = enh.image
    h, w = img.shape[:2]
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Ruling lines (tables, underlines, boxes) badly confuse OCR: find them,
    # remember the tables, then erase all rules from the OCR input.
    rules = detect_rules(enh.ink, dpi)
    tables = detect_ruled_tables(enh.ink, dpi, rules)
    frames = [(b, "boxed") for b in detect_rule_boxes(enh.ink, dpi, rules)]
    panels = getattr(enh, "panels", None)
    if panels is None:
        panels = detect_panels(getattr(enh, "gray", None), dpi)
    frames += [(tuple(map(float, b)), "shaded") for b in panels]
    rule_mask = cv2.dilate(cv2.bitwise_or(*rules), np.ones((3, 3), np.uint8))
    gray = gray.copy()
    if rule_mask.any():
        gray[rule_mask > 0] = 255
    # OCR sees panels as plain black-on-white (the tint is only cosmetic).
    for x0, y0, x1, y1 in (getattr(enh, "panels", None) or []):
        region = gray[y0:y1, x0:x1]
        region[enh.ink[y0:y1, x0:x1] == 0] = 255

    words: list[Word] = []
    has_structure = True
    if cfg.ocr:
        scale = cfg.ocr_dpi / float(dpi) if dpi != cfg.ocr_dpi else 1.0
        ocr_img = gray if scale == 1.0 else cv2.resize(
            gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
        engine = get_engine(cfg.ocr_engine)
        has_structure = engine.name == "tesseract"
        words = engine.recognize(ocr_img, cfg.ocr_lang)
        if scale != 1.0:
            for wd in words:
                wd.bbox = tuple(v / scale for v in wd.bbox)

    layout = analyze_layout(words, w, h, dpi, ink=enh.ink, figure_boxes=enh.figure_boxes,
                            has_structure=has_structure, tables=tables, frames=frames)

    # Code blocks: re-OCR verbatim on the monospace grid.
    for b in layout.blocks:
        if b.type == "code":
            tight = _union([l.bbox for l in b.lines]) if b.lines else b.bbox
            b.code = (reocr_code(gray, tight, cfg.ocr_lang) if cfg.ocr and has_structure else None) \
                or grid_text([wd for l in b.lines for wd in l.words])

    # Persist figure crops for Mode B (paths only travel back to the parent).
    from PIL import Image

    fig_i = 0
    for b in layout.blocks:
        if b.type != "figure":
            continue
        x0, y0, x1, y1 = (int(v) for v in b.bbox)
        crop = img[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]
        if crop.size == 0:
            continue
        path = f"{crop_prefix}_fig{fig_i}.jpg"
        Image.fromarray(crop).save(path, quality=88)
        b.image_path = path
        fig_i += 1
    return layout
