import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import synth  # noqa: E402


@pytest.fixture(scope="session")
def scanned_pdf(tmp_path_factory):
    """A small mixed 'bad scan': skews of both signs, a sideways page,
    a two-column page, a ruled table and a blank page."""
    specs = [
        {"kind": "single", "angle": -7.0},
        {"kind": "two_column", "angle": 3.5},
        {"kind": "table", "angle": 1.5},
        {"kind": "blank"},
        {"kind": "single", "angle": 0.0, "rotate90": 90},
    ]
    path = tmp_path_factory.mktemp("in") / "scan.pdf"
    path.write_bytes(synth.make_scanned_pdf(specs))
    return str(path)
