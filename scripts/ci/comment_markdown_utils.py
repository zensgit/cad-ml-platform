from __future__ import annotations

from typing import Iterable, Sequence


def stringify_cell(value: object, fallback: str = "") -> str:
    if value is None:
        return fallback
    return str(value)


def markdown_table(headers: Sequence[object], rows: Iterable[Sequence[object]]) -> str:
    normalized_headers = [stringify_cell(item) for item in headers]
    lines = [
        f"| {' | '.join(normalized_headers)} |",
        f"|{'|'.join('--------' for _ in normalized_headers)}|",
    ]
    for row in rows:
        lines.append(f"| {' | '.join(stringify_cell(item) for item in row)} |")
    return "\n".join(lines)


def markdown_section(title: str, body: str) -> str:
    return f"### {stringify_cell(title)}\n{stringify_cell(body)}"


def markdown_footer(*, updated_at: str = "", commit_sha: str) -> str:
    lines: list[str] = []
    timestamp = stringify_cell(updated_at).strip()
    if timestamp:
        lines.append(f"*Updated: {timestamp} UTC*")
    lines.append(f"*Commit: {stringify_cell(commit_sha)}*")
    return "\n".join(lines)
