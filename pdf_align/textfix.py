"""Post-OCR text clean-up for prose (never applied to code).

* Unicode normalisation: ligatures, smart quotes kept typographic, odd
  dashes and spaces unified.
* Line joining with de-hyphenation ("infor-" + "mation" -> "information").
* Conservative word repair:
    1. *Confusable-glyph* fixes that are almost always right, e.g. ``Smal!``
       -> ``Small`` (l/!), ``wi1l`` -> ``will`` (1/l), ``0ften`` -> ``often``
       - accepted only if the repaired word is in the dictionary and the
       original is not;
    2. a dictionary correction (edit distance 1) for words Tesseract itself
       was *unsure about* (confidence < 60), only when one candidate clearly
       dominates. Technical words with high OCR confidence are never touched.
"""

from __future__ import annotations

import functools
import logging
import re
import unicodedata

log = logging.getLogger(__name__)

_LIGATURES = {"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl",
              "ﬅ": "st", "ﬆ": "st"}
_NORMALISE = {" ": " ", " ": " ", " ": " ", " ": " ", "​": "",
              "‐": "-", "‑": "-", "‒": "-", "−": "-",
              "™": "", "­": ""}
_WORD_RE = re.compile(r"^([^\w]*)([\w'’!|]+?)([^\w]*)$", re.UNICODE)
_CONFUSABLE = [("!", "l"), ("|", "l"), ("|", "i"), ("|", "I"), ("1", "l"), ("1", "i"), ("0", "o"),
               ("5", "s"), ("I", "l"), ("rn", "m"), ("m", "rn"), ("vv", "w"), ("cl", "d"), ("li", "h"), ("l", "i"),
               ("l", "I")]


def normalize(text: str) -> str:
    for k, v in _LIGATURES.items():
        text = text.replace(k, v)
    for k, v in _NORMALISE.items():
        text = text.replace(k, v)
    text = unicodedata.normalize("NFC", text)
    return re.sub(r"[ \t]{2,}", " ", text)


def join_lines(lines: list[str]) -> str:
    """Join OCR lines into flowing text with de-hyphenation.

    'infor-' + 'mation' -> 'information'; but 'well-' + 'Known' and
    'pre-' + '2000' keep the hyphen.
    """
    out = ""
    for ln in (l.strip() for l in lines):
        if not ln:
            continue
        if not out:
            out = ln
        elif out.endswith("-") and len(out) > 1 and out[-2].isalpha() and ln[:1].islower():
            out = out[:-1] + ln
        else:
            out += " " + ln
    return re.sub(r"\s{2,}", " ", out)


# ---------------------------------------------------------------- dictionary
@functools.lru_cache(maxsize=1)
def _symspell():
    try:
        import importlib.resources as res

        from symspellpy import SymSpell

        sym = SymSpell(max_dictionary_edit_distance=1, prefix_length=7)
        path = res.files("symspellpy") / "frequency_dictionary_en_82_765.txt"
        sym.load_dictionary(str(path), term_index=0, count_index=1)
        return sym
    except Exception as exc:  # noqa: BLE001 - optional dependency
        log.info("spell fixing disabled: %s", exc)
        return None


def _count(word: str) -> int:
    sym = _symspell()
    return int(sym.words.get(word.lower(), 0)) if sym is not None else 0


def _known(word: str) -> bool:
    return _count(word) > 0


def _match_case(src: str, dst: str) -> str:
    if src.isupper() and len(src) > 1 and dst.isalpha():
        return dst.upper()
    if src[:1].isupper():
        return dst[:1].upper() + dst[1:].lower()
    return dst.lower() if dst.lower() != "i" else dst


def _confusable_candidates(word: str):
    """Variants with one confusable substitution (each occurrence separately,
    and all occurrences at once)."""
    for a, b in _CONFUSABLE:
        start = word.find(a)
        while start != -1:
            yield word[:start] + b + word[start + len(a):]
            start = word.find(a, start + 1)
        if word.count(a) > 1:
            yield word.replace(a, b)


def fix_word(token: str, conf: float = 100.0) -> str:
    """Repair a single OCR token (surrounding punctuation is preserved)."""
    if _symspell() is None:
        return token
    m = re.match(r"^([(\[{\"'“‘]*)(.*?)([)\]}\"'”’.,;:?]*)$", token)
    if not m:
        return token
    pre, core, post = m.groups()
    if not core or not any(c.isalpha() for c in core) or len(core) > 30:
        return token
    if _known(core) or not re.fullmatch(r"[A-Za-z0-9!|]+", core):
        return token  # known word, or identifier-like (dots, underscores, ...)
    # 1. confusable glyphs: pick the most frequent dictionary candidate.
    best = max((c for c in _confusable_candidates(core) if c.isalpha() and _known(c)),
               key=_count, default=None)
    if best:
        if core[:1] in "I|1" and best[:1].lower() in "li" and not core.isupper():
            return pre + best.lower() + post   # the capital-looking glyph *was* the error
        return pre + _match_case(core, best) + post
    # 2. unsure words: dictionary correction at edit distance 1.
    if conf < 60 and core.isalpha() and len(core) >= 4:
        from symspellpy import Verbosity

        sugg = _symspell().lookup(core.lower(), Verbosity.CLOSEST, max_edit_distance=1)
        if sugg and (len(sugg) == 1 or sugg[0].count >= 20 * sugg[1].count) and sugg[0].count > 50_000:
            return pre + _match_case(core, sugg[0].term) + post
    return token


def fix_line(words: list[tuple[str, float]], enabled: bool = True) -> str:
    """Join (text, confidence) word pairs of one line, repairing words."""
    if not enabled:
        return normalize(" ".join(t for t, _ in words))
    out = [fix_word(t, c) for t, c in words]
    # A "!" glued to a word and followed by a lower-case word is an OCR'd
    # "l" or stray mark, not an exclamation ("Smal! mistakes", "will! be").
    for i in range(len(out) - 1):
        w, nxt = out[i], out[i + 1]
        if w.endswith("!") and nxt[:1].islower() and len(w) > 2:
            core = w[:-1]
            out[i] = core + "l" if (not _known(core) and _known(core + "l")) else core
    return normalize(" ".join(out))
