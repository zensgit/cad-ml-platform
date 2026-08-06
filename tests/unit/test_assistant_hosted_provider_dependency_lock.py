"""Guard: `anthropic`/`openai` must not become installable assistant deps
before the provider SEAL lands.

`src/core/assistant/` is a mounted, CD-deployed service (`/assistant/*`)
whose `ClaudeProvider`/`OpenAIProvider` and `FunctionCallingEngine`'s
default `llm_provider="claude"` currently fail closed to an offline
fallback only because `anthropic`/`openai` are absent from every
`requirements*.txt` in this repo (verified in
docs/development/L3_ASSISTANT_SCOPE_SEAL_DESIGNLOCK_20260805.md, #535,
findings #6/#13/#15). That absence is presently the only thing preventing
a live, un-gated hosted-LLM call from a service that sits adjacent to
customer drawing data. This test locks the absence until the SEAL (#535
§2) is implemented, so a routine "add the SDK we already import" commit
cannot silently remove that protection.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REQUIREMENTS_FILES = sorted(ROOT.glob("requirements*.txt"))

# An uncommented line whose package name (before any version specifier)
# is exactly one of these. Matches `anthropic`, `anthropic==1.0`,
# `anthropic[extra]>=1.0`, but not a comment, not a substring hit inside
# another package name.
_LOCKED_PACKAGES = ("anthropic", "openai")
_PACKAGE_NAME_RE = re.compile(r"^([A-Za-z0-9_.-]+)")


def _active_package_names(path: Path) -> set[str]:
    names: set[str] = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        match = _PACKAGE_NAME_RE.match(line)
        if match:
            names.add(match.group(1).lower())
    return names


def test_requirements_files_are_discoverable_positive_control() -> None:
    """Positive control: the glob itself must find real files, or the
    absence assertions below would pass vacuously."""
    assert REQUIREMENTS_FILES, "no requirements*.txt found under repo root"
    assert (ROOT / "requirements.txt") in REQUIREMENTS_FILES


def test_requirements_files_positive_control_finds_known_package() -> None:
    """Positive control: prove `_active_package_names` actually parses
    active (uncommented) lines, using a package known to be present."""
    all_active: set[str] = set()
    for path in REQUIREMENTS_FILES:
        all_active |= _active_package_names(path)
    assert "fastapi" in all_active, (
        "expected 'fastapi' to be an active dependency somewhere; "
        "if this fails, the parser itself is broken, not the repo"
    )


def test_anthropic_and_openai_absent_from_every_requirements_file() -> None:
    """The locked invariant: no requirements*.txt may activate `anthropic`
    or `openai` before #535's SEAL lands."""
    offenders: dict[str, set[str]] = {}
    for path in REQUIREMENTS_FILES:
        hit = _active_package_names(path) & set(_LOCKED_PACKAGES)
        if hit:
            offenders[str(path.relative_to(ROOT))] = hit
    assert not offenders, (
        "anthropic/openai must not be added to any requirements*.txt "
        "before the assistant provider SEAL (#535, "
        "docs/development/L3_ASSISTANT_SCOPE_SEAL_DESIGNLOCK_20260805.md) "
        f"is implemented; found: {offenders}"
    )


def test_requirements_assistant_documents_the_lock() -> None:
    """The commented-out placeholder lines in requirements-assistant.txt
    must reference #535, so a future editor sees why before uncommenting."""
    path = ROOT / "requirements-assistant.txt"
    text = path.read_text(encoding="utf-8")
    assert "#535" in text or "L3_ASSISTANT_SCOPE_SEAL_DESIGNLOCK" in text, (
        "requirements-assistant.txt should explain, with a reference to "
        "#535, why anthropic/openai are commented out rather than simply "
        "absent"
    )
