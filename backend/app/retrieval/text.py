"""Tokenisation helpers shared by BM25, hashing embeddings and verification."""
from __future__ import annotations

import re

STOPWORDS = frozenset(
    """a an and are as at be been but by for from has have in into is it its of on or such that the their then
    there these this to was were which while with we our can may also using use used based than more most over under
    between within across both each other not no per via paper study results shown show shows however thus""".split()
)
_TOKEN = re.compile(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*")


def tokenize(text: str, keep_stop: bool = False) -> list[str]:
    toks = _TOKEN.findall(text.lower())
    out: list[str] = []
    for t in toks:
        out.append(t)
        if "-" in t or "_" in t:  # also index sub-parts: "ka-band" -> ka, band
            out.extend(p for p in re.split(r"[-_]", t) if p)
    return out if keep_stop else [t for t in out if t not in STOPWORDS and len(t) > 1]


def normalize(text: str) -> str:
    """Normalisation used for span matching (case, whitespace, ligatures, dashes)."""
    text = text.replace("ﬁ", "fi").replace("ﬂ", "fl").replace("–", "-").replace("—", "-")
    text = text.replace("‘", "'").replace("’", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", text.lower()).strip()
