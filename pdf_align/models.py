"""Serializable data model shared between workers, layout engine and builders.

Everything here round-trips through JSON so that workers can persist their
results to the disk-backed work directory and the parent process only keeps
lightweight handles in memory.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Literal, Optional

BBox = tuple[float, float, float, float]  # x0, y0, x1, y1 in image pixels

BlockType = Literal["heading", "paragraph", "table", "figure", "header", "footer", "list"]


@dataclass
class Word:
    text: str
    bbox: BBox
    conf: float = 100.0
    block: int = 0
    par: int = 0
    line: int = 0


@dataclass
class Line:
    words: list[Word]
    bbox: BBox

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]


@dataclass
class Block:
    type: BlockType
    bbox: BBox
    lines: list[Line] = field(default_factory=list)
    # For tables: rows -> cells -> text
    cells: Optional[list[list[str]]] = None
    # For figures: path of a cropped PNG on disk
    image_path: Optional[str] = None
    level: int = 0                       # heading level (1 = biggest)

    @property
    def text(self) -> str:
        return "\n".join(l.text for l in self.lines)


@dataclass
class PageLayout:
    width: int
    height: int
    blocks: list[Block] = field(default_factory=list)
    median_line_height: float = 0.0
    columns: int = 1
    # Every OCR word on the page (including words inside tables/figures that
    # are not part of the text flow). Used for Mode A's invisible text layer.
    ocr_words: list[Word] = field(default_factory=list)

    # ---- (de)serialisation ------------------------------------------------
    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @staticmethod
    def from_dict(d: dict) -> "PageLayout":
        blocks = []
        for b in d.get("blocks", []):
            lines = [
                Line(words=[Word(**{**w, "bbox": tuple(w["bbox"])}) for w in l["words"]],
                     bbox=tuple(l["bbox"]))
                for l in b.get("lines", [])
            ]
            blocks.append(
                Block(
                    type=b["type"],
                    bbox=tuple(b["bbox"]),
                    lines=lines,
                    cells=b.get("cells"),
                    image_path=b.get("image_path"),
                    level=b.get("level", 0),
                )
            )
        return PageLayout(
            width=d["width"],
            height=d["height"],
            blocks=blocks,
            median_line_height=d.get("median_line_height", 0.0),
            columns=d.get("columns", 1),
            ocr_words=[Word(**{**w, "bbox": tuple(w["bbox"])}) for w in d.get("ocr_words", [])],
        )

    @staticmethod
    def from_json(s: str) -> "PageLayout":
        return PageLayout.from_dict(json.loads(s))

    def all_words(self) -> list[Word]:
        return self.ocr_words or [w for b in self.blocks for l in b.lines for w in l.words]


PageStatus = Literal["ok", "blank", "error"]


@dataclass
class PageResult:
    """What a worker returns for one page. Only paths + scalars, never pixels."""

    index: int
    status: PageStatus = "ok"
    image_path: Optional[str] = None     # enhanced, deskewed page image (PNG/JPEG)
    layout_path: Optional[str] = None    # PageLayout JSON
    width_px: int = 0
    height_px: int = 0
    dpi: int = 300
    skew_deg: float = 0.0
    orientation_deg: int = 0
    message: str = ""
    seconds: float = 0.0

    def load_layout(self) -> Optional[PageLayout]:
        if not self.layout_path:
            return None
        with open(self.layout_path, "r", encoding="utf-8") as fh:
            return PageLayout.from_json(fh.read())


@dataclass
class Report:
    pages: list[PageResult] = field(default_factory=list)
    output_path: str = ""
    seconds: float = 0.0

    @property
    def total(self) -> int:
        return len(self.pages)

    @property
    def errors(self) -> list[PageResult]:
        return [p for p in self.pages if p.status == "error"]

    @property
    def blanks(self) -> list[PageResult]:
        return [p for p in self.pages if p.status == "blank"]

    @property
    def mean_abs_skew(self) -> float:
        ok = [abs(p.skew_deg) for p in self.pages if p.status == "ok"]
        return sum(ok) / len(ok) if ok else 0.0
