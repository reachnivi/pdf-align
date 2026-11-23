"""Book typesetting: Book model -> semantic HTML -> CSS paged media -> PDF."""

from .render import TRIMS, render_html, render_pdf, weasyprint_available

__all__ = ["TRIMS", "render_html", "render_pdf", "weasyprint_available"]
