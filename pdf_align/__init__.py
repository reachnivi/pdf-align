"""pdf_align — deskew, enhance and re-typeset poorly scanned PDFs.

Public entry points:

    from pdf_align import PipelineConfig, run_pipeline
    report = run_pipeline("scan.pdf", "clean.pdf", PipelineConfig(mode="A"))
"""

from .config import PipelineConfig

__all__ = ["PipelineConfig", "run_pipeline", "preview_pages"]
__version__ = "1.0.0"


def __getattr__(name):  # lazy: keep worker processes' import cost minimal
    if name in ("run_pipeline", "preview_pages"):
        from . import pipeline

        return getattr(pipeline, name)
    raise AttributeError(name)
