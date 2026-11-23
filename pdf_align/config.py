"""Pipeline configuration.

A single, picklable dataclass is passed to every worker process so that all
tuning knobs live in one place and the UI/CLI simply construct one of these.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from typing import Literal

Mode = Literal["A", "B"]
PageSize = Literal["original", "A4", "Letter"]
BlankPolicy = Literal["keep", "drop"]


@dataclass(frozen=True)
class PipelineConfig:
    # --- output ---------------------------------------------------------
    mode: Mode = "A"                    # A = visual deskew + OCR layer, B = full reflow
    page_size: PageSize = "original"    # Mode A: fit onto A4/Letter or keep scan size
    jpeg_quality: int = 80              # Mode A: image compression (0 => lossless PNG)

    # --- rasterisation --------------------------------------------------
    dpi: int = 300

    # --- computer vision ------------------------------------------------
    max_skew_deg: float = 15.0          # search window for skew detection
    detect_orientation: bool = True     # fix 90/180/270 rotated pages
    enhance: bool = True
    binarize_text: bool = True          # adaptive threshold on text regions
    enhance_strength: float = 1.0       # 0.5 (gentle) .. 2.0 (aggressive)
    blank_policy: BlankPolicy = "keep"

    # --- OCR ------------------------------------------------------------
    ocr: bool = True
    ocr_engine: str = "tesseract"       # "tesseract" | "surya"
    ocr_lang: str = "eng"
    ocr_dpi: int = 300                  # OCR is run on an image at this DPI

    # --- performance ----------------------------------------------------
    batch_size: int = 8
    workers: int = max(1, (os.cpu_count() or 2) - 1)
    page_timeout_s: int = 180

    def to_dict(self) -> dict:
        return asdict(self)

    def validated(self) -> "PipelineConfig":
        """Return a copy with values clamped into sane ranges."""
        from dataclasses import replace

        return replace(
            self,
            dpi=int(min(max(self.dpi, 72), 600)),
            ocr_dpi=int(min(max(self.ocr_dpi, 150), 400)),
            batch_size=int(min(max(self.batch_size, 1), 64)),
            workers=int(min(max(self.workers, 1), os.cpu_count() or 1)),
            max_skew_deg=float(min(max(self.max_skew_deg, 1.0), 45.0)),
            enhance_strength=float(min(max(self.enhance_strength, 0.25), 3.0)),
            jpeg_quality=int(min(max(self.jpeg_quality, 0), 100)),
        )
