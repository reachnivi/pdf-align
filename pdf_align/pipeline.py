"""Streaming, multi-process orchestration.

Memory model
------------
* The input PDF is never rasterised as a whole. Each worker opens the file
  itself and renders exactly one page.
* At most ``batch_size`` pages are in flight at any moment (sliding window).
  A finished page exists only as a small PNG/JPEG + JSON in the work dir.
* Results are fed to the builder strictly in page order through a small
  re-order buffer, and Mode A deletes each page image right after use.
* The work dir is a ``TemporaryDirectory`` (context manager), so it is removed
  on success, on error and on interpreter exit.

Peak RSS ~= workers x (one page at ``dpi``) + the builder's flushed document.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import re
import sys
import tempfile
import time
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from concurrent.futures import TimeoutError as FutureTimeout  # distinct from builtin on 3.10
from concurrent.futures.process import BrokenProcessPool
from dataclasses import replace
from typing import Callable, Iterable, Optional

import pymupdf as fitz

from .config import PipelineConfig
from .deskew import PageTask, process_page, render_page, worker_init
from .models import PageResult, Report
from .pdf_builder import make_builder

log = logging.getLogger(__name__)

# progress_cb(done, total, stage_message, page_result_or_None)
ProgressCB = Callable[[int, int, str, Optional[PageResult]], None]


class PDFInputError(ValueError):
    """The input file cannot be used at all (encrypted, not a PDF, no pages)."""


# =============================================================================
# Input validation / repair
# =============================================================================

def inspect_pdf(path: str, work_dir: str) -> tuple[str, int]:
    """Validate the input and return ``(usable_path, page_count)``.

    If MuPDF had to repair the file while opening it (broken xref etc.) a
    cleaned copy is written to the work dir so that every worker doesn't
    repeat the repair.
    """
    fitz.TOOLS.mupdf_warnings()  # reset
    try:
        doc = fitz.open(path)
    except Exception as exc:  # noqa: BLE001
        raise PDFInputError(f"Not a readable PDF: {exc}") from exc
    try:
        if doc.needs_pass:
            raise PDFInputError("The PDF is password-protected. Please upload an unlocked copy.")
        if not doc.is_pdf:
            raise PDFInputError("The file is not a PDF document.")
        n = doc.page_count
        if n == 0:
            raise PDFInputError("The PDF has no pages.")
        warnings = fitz.TOOLS.mupdf_warnings()
        if doc.is_repaired or warnings:
            repaired = os.path.join(work_dir, "_repaired_input.pdf")
            log.warning("input needed repair (%s); writing cleaned copy", (warnings or "")[:200])
            doc.save(repaired, garbage=4, deflate=True)
            return repaired, n
        return path, n
    finally:
        doc.close()


# =============================================================================
# Execution backends
# =============================================================================

def _iter_serial(tasks: list[PageTask]) -> Iterable[PageResult]:
    for t in tasks:
        yield process_page(t)


def _error_result(task: PageTask, msg: str) -> PageResult:
    return PageResult(index=task.index, status="error", dpi=task.config.dpi, message=msg)


# Recycle each worker after this many pages: large NumPy/OpenCV buffers
# fragment the C heap, and a fresh process returns all of it to the OS.
MAX_PAGES_PER_WORKER = 12


def _new_pool(workers: int) -> ProcessPoolExecutor:
    # "spawn" is the only start method that is safe with MuPDF/OpenCV native
    # state and with threaded hosts such as Streamlit's script runner.
    kwargs = {}
    if sys.version_info >= (3, 11):
        kwargs["max_tasks_per_child"] = MAX_PAGES_PER_WORKER
    return ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"),
                               initializer=worker_init, **kwargs)


def _kill_pool(pool: ProcessPoolExecutor) -> None:
    """Hard-stop a pool that contains a hung worker."""
    procs = list(getattr(pool, "_processes", {}).values())
    pool.shutdown(wait=False, cancel_futures=True)
    for p in procs:
        try:
            p.terminate()
        except Exception:  # noqa: BLE001
            pass


def _run_isolated(task: PageTask, timeout: float) -> PageResult:
    """Run one page in its own single-worker process.

    Used to retry pages after a pool crash: if *this* page is the one that
    kills its process (segfault in a native lib, OOM), it is reported as an
    error page instead of taking the whole job (or the UI process) down.
    """
    pool = _new_pool(1)
    try:
        return pool.submit(process_page, task).result(timeout=timeout)
    except BrokenProcessPool:
        return _error_result(task, "page crashed the worker process (corrupt or oversized content)")
    except FutureTimeout:
        return _error_result(task, f"timed out after {timeout:.0f}s")
    except Exception as exc:  # noqa: BLE001
        return _error_result(task, f"worker error: {exc}")
    finally:
        _kill_pool(pool)


def _iter_parallel(tasks: list[PageTask], cfg: PipelineConfig) -> Iterable[PageResult]:
    """Yield results as they complete, keeping <= batch_size pages in flight.

    Robustness:
      * per-page timeout -> that page becomes an error page and the pool is
        recycled (the only way to stop a hung native call);
      * crashed worker (BrokenProcessPool, e.g. OOM-kill or a segfault inside
        a native library) -> the pool is recycled and every page that was in
        flight is retried once in an isolated process.
    """
    queue = list(reversed(tasks))  # pop() from the end = page order
    window = max(cfg.batch_size, cfg.workers)
    pool = _new_pool(cfg.workers)
    inflight: dict[Future, tuple[PageTask, float]] = {}
    try:
        while queue or inflight:
            while queue and len(inflight) < window:
                t = queue.pop()
                inflight[pool.submit(process_page, t)] = (t, time.monotonic())

            done, _ = wait(list(inflight), timeout=2.0, return_when=FIRST_COMPLETED)
            crashed: list[PageTask] = []
            for fut in done:
                task, _ = inflight.pop(fut)
                try:
                    yield fut.result()
                except BrokenProcessPool:
                    crashed.append(task)
                except Exception as exc:  # noqa: BLE001
                    yield _error_result(task, f"worker error: {exc}")

            now = time.monotonic()
            expired = [f for f, (_, t0) in inflight.items() if not f.done() and now - t0 > cfg.page_timeout_s]
            if not crashed and not expired:
                continue

            for f in expired:
                task, _ = inflight.pop(f)
                yield _error_result(task, f"timed out after {cfg.page_timeout_s}s")
            survivors = [task for task, _ in inflight.values()]
            inflight.clear()
            _kill_pool(pool)
            if crashed:
                log.warning("worker pool crashed; retrying %d page(s) in isolation",
                            len(crashed) + len(survivors))
                for task in crashed + survivors:
                    yield _run_isolated(task, cfg.page_timeout_s)
            else:
                queue.extend(reversed(survivors))  # innocent bystanders: rerun
            pool = _new_pool(cfg.workers)
    finally:
        _kill_pool(pool)


# =============================================================================
# Public API
# =============================================================================

def run_pipeline(input_path: str, output_path: str, config: Optional[PipelineConfig] = None,
                 progress_cb: Optional[ProgressCB] = None,
                 pages: Optional[list[int]] = None) -> Report:
    """Process ``input_path`` into ``output_path``. Returns a per-page Report.

    ``pages`` optionally restricts processing to a subset (0-based indexes).
    """
    cfg = (config or PipelineConfig()).validated()
    t_start = time.perf_counter()
    cb = progress_cb or (lambda *a: None)
    if cfg.mode == "B" and not cfg.title:
        cfg = replace(cfg, title=guess_title(input_path))

    with tempfile.TemporaryDirectory(prefix="pdfalign_") as work_dir:
        src, n_pages = inspect_pdf(input_path, work_dir)
        indices = [i for i in (pages if pages is not None else range(n_pages)) if 0 <= i < n_pages]
        total = len(indices)
        cb(0, total, f"Starting: {total} page(s), {cfg.workers} worker(s)", None)

        tasks = [PageTask(src, i, work_dir, cfg) for i in indices]
        stream = _iter_serial(tasks) if cfg.workers <= 1 or total <= 1 else _iter_parallel(tasks, cfg)

        builder = make_builder(cfg, output_path, work_dir)
        report = Report(output_path=output_path)
        reorder: dict[int, PageResult] = {}
        order = iter(indices)
        next_idx = next(order, None)
        done = 0

        for res in stream:
            done += 1
            reorder[res.index] = res
            label = {"ok": f"skew {res.skew_deg:+.2f}°" + (f", rotated {res.orientation_deg}°" if res.orientation_deg else ""),
                     "blank": "blank page", "error": f"ERROR: {res.message[:80]}"}[res.status]
            cb(done, total, f"Page {res.index + 1}: {label}", res)
            # Emit to the builder strictly in page order.
            while next_idx is not None and next_idx in reorder:
                r = reorder.pop(next_idx)
                builder.add_page(r)
                report.pages.append(r)
                if cfg.mode == "A":
                    _cleanup_page_files(r)
                next_idx = next(order, None)

        cb(total, total, "Typesetting the book…" if cfg.mode == "B" else "Assembling final PDF…", None)
        builder.close()
        report.warning = getattr(builder, "warning", "")
    # work_dir (all page images / JSON / partial PDFs) is gone at this point.
    report.seconds = time.perf_counter() - t_start
    cb(total, total, f"Done in {report.seconds:.1f}s", None)
    return report


def guess_title(path: str) -> str:
    """Book title from PDF metadata, else a tidied file name."""
    try:
        with fitz.open(path) as d:
            t = (d.metadata or {}).get("title", "").strip()
        if t and len(t) > 3 and not t.lower().endswith((".pdf", ".tif", ".doc")) and "untitled" not in t.lower():
            return t
    except Exception:  # noqa: BLE001
        pass
    stem = os.path.splitext(os.path.basename(path))[0]
    stem = re.sub(r"^in_[0-9a-f]{16}$", "", stem)  # UI temp names
    stem = re.sub(r"[_\-.]+", " ", stem).strip()
    stem = re.sub(r"\b(scan(ned)?|ocr|final|v\d+)\b", "", stem, flags=re.I).strip()
    return stem.title() if stem else "Untitled"


def _cleanup_page_files(r: PageResult) -> None:
    for p in (r.image_path, r.layout_path):
        if p and os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass


def preview_pages(input_path: str, config: Optional[PipelineConfig] = None,
                  pages: tuple[int, ...] = (0, 1), preview_dpi: int = 110) -> list[dict]:
    """Before/after previews for a few pages (used by the UI).

    Returns ``[{"index", "before": PNG bytes, "after": PNG bytes, "result": PageResult}]``.
    "After" is rendered from a real mini output PDF in the selected mode, so
    the preview is exactly what the final document will look like.
    """
    cfg = (config or PipelineConfig()).validated()
    out = []
    with tempfile.TemporaryDirectory(prefix="pdfalign_prev_") as wd:
        src, n = inspect_pdf(input_path, wd)
        idx = [p for p in pages if p < n]
        mini = os.path.join(wd, "preview.pdf")
        report = run_pipeline(src, mini, cfg, pages=idx)
        with fitz.open(mini) as doc:
            after_pages = [doc[i].get_pixmap(dpi=preview_dpi).tobytes("png") for i in range(min(doc.page_count, 4))]
        for k, i in enumerate(idx):
            before_rgb, _ = render_page(src, i, preview_dpi)
            import cv2

            ok, enc = cv2.imencode(".png", cv2.cvtColor(before_rgb, cv2.COLOR_RGB2BGR))
            # Mode B reflows text, so output page k is not necessarily input
            # page k; show the first output pages in order.
            out.append({"index": i, "before": enc.tobytes(),
                        "after": after_pages[k] if k < len(after_pages) else None,
                        "result": report.pages[k] if k < len(report.pages) else None})
    return out
