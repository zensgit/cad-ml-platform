from __future__ import annotations

import re
from typing import Any, List, Sequence

_WS_RE = re.compile(r"\s+")
_MTEXT_STACK_RE = re.compile(r"\\\\S([^;]+);")
_MTEXT_FMT_RE = re.compile(r"\\\\[A-Za-z][^;]*;")
_MTEXT_CODE_RE = re.compile(r"\\\\[A-Za-z]")


def _clean_text(raw: str, *, is_mtext: bool) -> str:
    text = str(raw or "")
    if not text:
        return ""

    if is_mtext:
        # Common MTEXT control sequences:
        # - \P paragraph/newline
        # - {\H...; ...} formatting groups
        text = text.replace("\\P", " ").replace("\\n", " ").replace("\\r", " ")
        text = text.replace("{", " ").replace("}", " ")
        # Stacked fractions: \S1^2; or \S1#2; -> "1/2"
        text = _MTEXT_STACK_RE.sub(lambda m: m.group(1).replace("^", "/").replace("#", "/"), text)
        # Formatting runs: \H0.7x; \C1; \fArial|b0|i0; ...
        text = _MTEXT_FMT_RE.sub(" ", text)
        # Remaining single-letter toggles: \L \l \O ...
        text = _MTEXT_CODE_RE.sub(" ", text)

    # AutoCAD %% codes (seen in both TEXT and MTEXT).
    text = re.sub(r"%%[cC]", " dia ", text)
    text = re.sub(r"%%[dD]", " deg ", text)
    text = re.sub(r"%%[pP]", " pm ", text)

    text = text.strip()
    text = _WS_RE.sub(" ", text)
    return text.lower()


def parse_text_content(entities: Sequence[Any], *, max_items: int = 5000) -> List[str]:
    """Extract normalized text tokens from DXF modelspace entities.

    Output is a sorted list of strings to make downstream JSON comparison stable.
    """
    out: List[str] = []
    budget = max(0, int(max_items))
    if budget <= 0:
        return out

    for ent in entities:
        try:
            et = ent.dxftype()
        except Exception:
            continue
        if et not in {"TEXT", "MTEXT"}:
            continue

        raw = ""
        if et == "MTEXT":
            # Prefer ezdxf's plain_text() when available to strip formatting.
            try:
                if hasattr(ent, "plain_text"):
                    raw = ent.plain_text()  # type: ignore[call-arg]
                else:
                    raw = getattr(ent, "text", "") or getattr(ent.dxf, "text", "")
            except Exception:
                raw = getattr(ent, "text", "") or getattr(ent.dxf, "text", "")
        else:
            try:
                raw = getattr(ent.dxf, "text", "") or ""
            except Exception:
                raw = ""

        cleaned = _clean_text(raw, is_mtext=(et == "MTEXT"))
        if not cleaned:
            continue
        out.append(cleaned)
        if len(out) >= budget:
            break

    out.sort()
    return out

