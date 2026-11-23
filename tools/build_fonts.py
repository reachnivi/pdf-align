"""Fetch the open-source (SIL OFL 1.1) book fonts and freeze static instances.

    python tools/build_fonts.py

Google Fonts distributes Source Serif 4 / Source Sans 3 / Source Code Pro as
variable fonts. Static instances (Regular, Italic, Semibold, Bold) embed and
subset reliably in every PDF engine, so we instantiate them once here and
commit the results to pdf_align/typeset/fonts/ together with the licences.
"""

from __future__ import annotations

import io
import os
import urllib.request

from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont

BASE = "https://raw.githubusercontent.com/google/fonts/main/ofl/"
OUT = os.path.join(os.path.dirname(__file__), "..", "pdf_align", "typeset", "fonts")

FAMILIES = {
    # family dir: (file name, {style: axis values}, output prefix)
    "sourceserif4": [
        ("SourceSerif4[opsz,wght].ttf", {"Regular": {"wght": 400, "opsz": 10}, "Semibold": {"wght": 600, "opsz": 10},
                                         "Bold": {"wght": 700, "opsz": 10}}, "SourceSerif4"),
        ("SourceSerif4-Italic[opsz,wght].ttf", {"Italic": {"wght": 400, "opsz": 10},
                                                "BoldItalic": {"wght": 700, "opsz": 10}}, "SourceSerif4"),
    ],
    "sourcesans3": [
        ("SourceSans3[wght].ttf", {"Regular": {"wght": 400}, "Semibold": {"wght": 600},
                                   "Bold": {"wght": 700}}, "SourceSans3"),
        ("SourceSans3-Italic[wght].ttf", {"Italic": {"wght": 400}}, "SourceSans3"),
    ],
    "sourcecodepro": [
        ("SourceCodePro[wght].ttf", {"Regular": {"wght": 400}, "Semibold": {"wght": 600},
                                     "Bold": {"wght": 700}}, "SourceCodePro"),
        ("SourceCodePro-Italic[wght].ttf", {"Italic": {"wght": 400}}, "SourceCodePro"),
    ],
}


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read()


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    for fam, files in FAMILIES.items():
        with open(os.path.join(OUT, f"OFL-{fam}.txt"), "wb") as fh:
            fh.write(fetch(BASE + fam + "/OFL.txt"))
        for fname, styles, prefix in files:
            data = fetch(BASE + fam + "/" + urllib.request.quote(fname))
            for style, axes in styles.items():
                font = TTFont(io.BytesIO(data))
                axes = {k: v for k, v in axes.items() if k in {a.axisTag for a in font["fvar"].axes}}
                static = instantiateVariableFont(font, axes, updateFontNames=True)
                path = os.path.join(OUT, f"{prefix}-{style}.ttf")
                static.save(path)
                print(f"{path}  {os.path.getsize(path) // 1024} KB")


if __name__ == "__main__":
    main()
