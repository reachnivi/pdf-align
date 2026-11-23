"""Font discovery across platforms (simulated - no real filesystem needed)."""

import os

from pdf_align import pdf_builder


def _fake_fs(monkeypatch, existing):
    existing = {os.path.expanduser(p) for p in existing}
    monkeypatch.setattr(pdf_builder.os.path, "exists", lambda p: p in existing)


def test_mac_homebrew_dejavu_in_user_library(monkeypatch):
    _fake_fs(monkeypatch, ["~/Library/Fonts/DejaVuSans.ttf", "~/Library/Fonts/DejaVuSans-Bold.ttf",
                           "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"])
    assert pdf_builder.find_unicode_font() == os.path.expanduser("~/Library/Fonts/DejaVuSans.ttf")
    assert pdf_builder.find_unicode_font(bold=True) == os.path.expanduser("~/Library/Fonts/DejaVuSans-Bold.ttf")


def test_mac_builtin_fallback_has_real_bold(monkeypatch):
    _fake_fs(monkeypatch, ["/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
                           "/System/Library/Fonts/Supplemental/Arial Bold.ttf"])
    assert pdf_builder.find_unicode_font().endswith("Arial Unicode.ttf")
    assert pdf_builder.find_unicode_font(bold=True).endswith("Arial Bold.ttf")


def test_linux_dejavu(monkeypatch):
    _fake_fs(monkeypatch, ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                           "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"])
    assert pdf_builder.find_unicode_font(bold=True).endswith("DejaVuSans-Bold.ttf")


def test_no_fonts_falls_back_to_none(monkeypatch):
    _fake_fs(monkeypatch, [])
    assert pdf_builder.find_unicode_font() is None
