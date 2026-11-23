"""Command-line interface (headless / batch use).

    python -m pdf_align.cli scan.pdf clean.pdf --mode A --dpi 300 --workers 4
"""

from __future__ import annotations

import argparse
import logging
import sys

from .config import PipelineConfig
from .pipeline import PDFInputError, run_pipeline


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="pdf-align", description="Deskew, enhance and re-typeset scanned PDFs.")
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--mode", choices=["A", "B"], default="A",
                    help="A = visual deskew + invisible OCR layer; B = full reflow / re-typeset")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--page-size", choices=["original", "A4", "Letter"], default="original",
                    help="Mode A page size")
    ap.add_argument("--trim", choices=["7x9.19in", "7.375x9.25in", "A4", "Letter"], default="7x9.19in",
                    help="Mode B book trim size")
    ap.add_argument("--title", default="", help="Mode B book title (default: PDF metadata / file name)")
    ap.add_argument("--author", default="")
    ap.add_argument("--chapters-any-page", action="store_true",
                    help="Mode B: start chapters on the next page instead of the next right-hand page")
    ap.add_argument("--no-front-matter", action="store_true", help="Mode B: no title page / contents")
    ap.add_argument("--no-spellfix", action="store_true", help="disable OCR word repair")
    ap.add_argument("--tessdata-dir", default="", help="folder with Tesseract models (e.g. tessdata_best)")
    ap.add_argument("--emit-html", default="", help="Mode B: also write the intermediate HTML here")
    ap.add_argument("--lang", default="eng", help="Tesseract language(s), e.g. 'eng+deu'")
    ap.add_argument("--engine", choices=["tesseract", "surya"], default="tesseract")
    ap.add_argument("--workers", type=int, default=PipelineConfig().workers)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--strength", type=float, default=1.0, help="enhancement strength 0.25-3")
    ap.add_argument("--no-ocr", action="store_true")
    ap.add_argument("--no-binarize", action="store_true", help="keep grayscale text instead of pure B/W")
    ap.add_argument("--drop-blank", action="store_true")
    ap.add_argument("--jpeg-quality", type=int, default=80)
    ap.add_argument("-q", "--quiet", action="store_true")
    a = ap.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    cfg = PipelineConfig(
        mode=a.mode, dpi=a.dpi, page_size=a.page_size, ocr_lang=a.lang, ocr_engine=a.engine,
        workers=a.workers, batch_size=a.batch_size, enhance_strength=a.strength, ocr=not a.no_ocr,
        binarize_text=not a.no_binarize, blank_policy="drop" if a.drop_blank else "keep",
        jpeg_quality=a.jpeg_quality, trim=a.trim, title=a.title, author=a.author,
        chapters_on_right=not a.chapters_any_page, front_matter=not a.no_front_matter,
        spellfix=not a.no_spellfix, tessdata_dir=a.tessdata_dir, html_out=a.emit_html,
    )

    def progress(done, total, msg, _res):
        if not a.quiet:
            print(f"\r[{done:>4}/{total}] {msg[:90]:<90}", end="" if done < total else "\n", flush=True)

    try:
        rep = run_pipeline(a.input, a.output, cfg, progress)
    except PDFInputError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"\nWrote {a.output}: {rep.total} pages in {rep.seconds:.1f}s "
          f"({rep.seconds / max(rep.total, 1):.2f}s/page), mean |skew| {rep.mean_abs_skew:.2f}°, "
          f"{len(rep.blanks)} blank, {len(rep.errors)} error(s)")
    if rep.warning:
        print(f"warning: {rep.warning}", file=sys.stderr)
    for p in rep.errors:
        print(f"  page {p.index + 1}: {p.message}", file=sys.stderr)
    return 1 if rep.errors and len(rep.errors) == rep.total else 0


if __name__ == "__main__":
    sys.exit(main())
