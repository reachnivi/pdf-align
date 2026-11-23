"""PDF-Align — Streamlit web UI.

    streamlit run app.py

Drag & drop a scanned PDF, preview the first two pages before/after, then
process the whole document with a live "Page X of Y" progress bar and
download the result.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import textwrap
import time
from dataclasses import asdict

import streamlit as st

from pdf_align.config import PipelineConfig
from pdf_align.layout_engine import tesseract_available
from pdf_align.pipeline import PDFInputError, preview_pages, run_pipeline


@st.cache_data(show_spinner=False, max_entries=6)
def _preview(path: str, digest: str, cfg_dict: dict):
    preview_cfg = PipelineConfig(**{**cfg_dict, "workers": min(2, cfg_dict["workers"])})
    return preview_pages(path, preview_cfg, pages=(0, 1))


def main() -> None:
    st.set_page_config(page_title="PDF-Align · Scan Cleaner", page_icon="📄", layout="wide")

    st.markdown(
        textwrap.dedent("""
        <style>
          .block-container {padding-top: 2rem; max-width: 1300px;}
          .pa-title {font-size: 2.1rem; font-weight: 700; margin-bottom: .1rem;}
          .pa-sub {color: #6b7280; margin-bottom: 1.4rem;}
          .pa-cap {text-align:center; color:#6b7280; font-size:.85rem; margin-top:.25rem;}
          div[data-testid="stImage"] img {border: 1px solid rgba(128,128,128,.25); border-radius: 6px;
                                           box-shadow: 0 2px 10px rgba(0,0,0,.08);}
        </style>
        <div class="pa-title">📄 PDF-Align</div>
        <div class="pa-sub">Straighten, clean and re-typeset badly scanned PDFs — even 100s of pages.</div>
        """),
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------------------------
    # Session-scoped scratch directory (uploads + outputs)
    # --------------------------------------------------------------------------
    if "tmpdir" not in st.session_state:
        st.session_state.tmpdir = tempfile.TemporaryDirectory(prefix="pdfalign_ui_")
    TMP = st.session_state.tmpdir.name

    tess_ok, tess_info = tesseract_available()

    # --------------------------------------------------------------------------
    # Sidebar: settings
    # --------------------------------------------------------------------------
    with st.sidebar:
        st.header("⚙️ Settings")
        mode_label = st.radio(
            "Output mode",
            ["A · Visual deskew & enhance", "B · Full reflow & re-typeset"],
            help="A keeps the original page look (straightened, whitened) with an invisible, searchable "
                 "OCR text layer. B rebuilds the document with clean fonts, justified paragraphs and real tables.",
        )
        mode = "A" if mode_label.startswith("A") else "B"

        page_size = st.selectbox(
            "Output page size",
            ["original", "A4", "Letter"] if mode == "A" else ["A4", "Letter"],
            help="Mode A can keep each scan's size or fit pages onto a standard size.",
        )
        dpi = st.select_slider("Scan resolution (DPI)", options=[150, 200, 300, 400], value=300)
        strength = st.slider("Enhancement strength", 0.25, 2.0, 1.0, 0.25,
                             help="Higher = whiter paper and darker faded text.")
        binarize = st.toggle("Crisp black & white text", value=True,
                             help="Off keeps gray-scale anti-aliased text.")

        st.subheader("OCR")
        ocr = st.toggle("Run OCR (searchable text)", value=tess_ok or mode == "B", disabled=not tess_ok)
        lang = st.text_input("Tesseract language(s)", "eng", help="e.g. 'eng', 'deu', 'eng+fra'")
        if not tess_ok:
            st.error("Tesseract not found — OCR & Mode B are unavailable. See README for installation.")

        with st.expander("Performance & pages"):
            cpu = os.cpu_count() or 2
            workers = st.slider("Worker processes", 1, cpu, max(1, cpu - 1))
            batch = st.slider("Pages in flight (batch size)", 1, 32, 8,
                              help="Bounds memory: only this many pages are being processed at once.")
            drop_blank = st.toggle("Remove blank pages", value=False)
            timeout = st.number_input("Per-page timeout (s)", 30, 1200, 180, step=30)

    cfg = PipelineConfig(
        mode=mode, page_size=page_size, dpi=dpi, enhance_strength=strength, binarize_text=binarize,
        ocr=ocr and tess_ok, ocr_lang=lang.strip() or "eng", workers=workers, batch_size=batch,
        blank_policy="drop" if drop_blank else "keep", page_timeout_s=int(timeout),
    )
    if mode == "B" and not cfg.ocr:
        st.warning("Mode B needs OCR to rebuild text. Enable OCR or switch to Mode A.")

    # --------------------------------------------------------------------------
    # Upload
    # --------------------------------------------------------------------------
    uploaded = st.file_uploader("Drop a scanned PDF here", type=["pdf"], accept_multiple_files=False)
    if uploaded is None:
        st.info("⬆️ Upload a PDF to begin. Processing happens locally on this server; "
                "files are deleted when your session ends.")
        return

    # Persist the upload to disk once (keyed by content hash) - the pipeline
    # streams pages from disk and never needs the whole file in memory again.
    data = uploaded.getvalue()
    digest = hashlib.sha1(data).hexdigest()[:16]
    in_path = os.path.join(TMP, f"in_{digest}.pdf")
    if not os.path.exists(in_path):
        with open(in_path, "wb") as fh:
            fh.write(data)
    del data

    try:
        import pymupdf

        with pymupdf.open(in_path) as _d:
            n_pages = _d.page_count
            encrypted = _d.needs_pass
    except Exception as exc:  # noqa: BLE001
        st.error(f"This file could not be opened as a PDF: {exc}")
        return
    if encrypted:
        st.error("This PDF is password-protected. Please upload an unlocked copy.")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("File", uploaded.name if len(uploaded.name) < 28 else uploaded.name[:25] + "…")
    c2.metric("Pages", n_pages)
    c3.metric("Size", f"{uploaded.size / 1e6:.1f} MB")

    # --------------------------------------------------------------------------
    # Before / after preview of the first two pages
    # --------------------------------------------------------------------------
    st.subheader("👀 Preview — before vs. after")
    try:
        with st.spinner("Rendering preview of the first pages…"):
            previews = _preview(in_path, digest, asdict(cfg))
    except PDFInputError as exc:
        st.error(str(exc))
        return
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Preview failed ({exc}); you can still process the document.")
        previews = []

    for k, pv in enumerate(previews):
        left, right = st.columns(2)
        res = pv["result"]
        with left:
            st.image(pv["before"], width="stretch")
            st.markdown(f"<div class='pa-cap'>Original · page {pv['index'] + 1}</div>", unsafe_allow_html=True)
        with right:
            if pv["after"] is not None:
                st.image(pv["after"], width="stretch")
            detail = ""
            if res is not None:
                detail = {"ok": f"skew corrected {res.skew_deg:+.2f}°"
                                + (f", rotated {res.orientation_deg}°" if res.orientation_deg else ""),
                          "blank": "blank page", "error": f"⚠️ {res.message[:80]}"}[res.status]
            label = f"Processed · page {pv['index'] + 1}" if mode == "A" else f"Re-typeset · output page {k + 1}"
            st.markdown(f"<div class='pa-cap'>{label} · {detail}</div>", unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # Full processing
    # --------------------------------------------------------------------------
    st.subheader("🚀 Process the full document")
    result_key = f"{digest}:{hashlib.sha1(repr(asdict(cfg)).encode()).hexdigest()[:10]}"
    go = st.button(f"Process all {n_pages} pages", type="primary", width="stretch")

    if go:
        out_path = os.path.join(TMP, f"out_{result_key.replace(':', '_')}.pdf")
        bar = st.progress(0.0, text="Starting…")
        status = st.empty()
        log_box = st.expander("Live log", expanded=False)
        log_area = log_box.empty()
        log_lines: list[str] = []
        t0 = time.time()

        def on_progress(done: int, total: int, msg: str, _res):
            frac = done / total if total else 1.0
            eta = ""
            if 0 < done < total:
                eta = f" · ~{(time.time() - t0) / done * (total - done):.0f}s left"
            bar.progress(min(frac, 1.0), text=f"Page {done} of {total}{eta}")
            status.caption(msg)
            log_lines.append(msg)
            log_area.code("\n".join(log_lines[-40:]), language=None)

        try:
            report = run_pipeline(in_path, out_path, cfg, on_progress)
            st.session_state.result = {"key": result_key, "path": out_path, "report": report,
                                       "name": os.path.splitext(uploaded.name)[0] + f"_aligned_{mode}.pdf"}
            bar.progress(1.0, text=f"Done — {report.total} pages in {report.seconds:.1f}s")
        except PDFInputError as exc:
            st.error(str(exc))
        except Exception as exc:  # noqa: BLE001
            st.exception(exc)

    res = st.session_state.get("result")
    if res and res["key"] == result_key and os.path.exists(res["path"]):
        rep = res["report"]
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Pages", rep.total)
        m2.metric("Mean |skew| fixed", f"{rep.mean_abs_skew:.2f}°")
        m3.metric("Blank pages", len(rep.blanks))
        m4.metric("Errors", len(rep.errors))
        m5.metric("Time", f"{rep.seconds:.1f}s")
        if rep.errors:
            st.warning("Some pages could not be fully processed; they were kept as original images or "
                       "placeholders:\n\n" + "\n".join(f"- page {p.index + 1}: {p.message}" for p in rep.errors[:20]))
        with open(res["path"], "rb") as fh:
            st.download_button("⬇️ Download cleaned PDF", fh, file_name=res["name"], mime="application/pdf",
                               type="primary", width="stretch")
        with st.expander("Per-page details"):
            st.dataframe(
                [{"page": p.index + 1, "status": p.status, "skew°": p.skew_deg, "rotated°": p.orientation_deg,
                  "seconds": round(p.seconds, 2), "note": p.message} for p in rep.pages],
                width="stretch", hide_index=True,
            )


# Streamlit executes this script as ``__main__``. Worker processes started with
# the "spawn" method re-import the main module as ``__mp_main__`` - the guard
# keeps them from re-running the UI.
if __name__ == "__main__":
    main()
